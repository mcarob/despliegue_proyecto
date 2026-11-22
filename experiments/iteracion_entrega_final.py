"""Iteración de modelos de la entrega final: ¿se puede mejorar el random forest de la Entrega 2?

Corrige una debilidad de la Entrega 2 (los modelos se compararon directamente en la prueba 2022-2024) y prueba
mejoras con una validación temporal móvil: para 2019, 2020 y 2021 se entrena con los años anteriores y se predice
ese año. La prueba 2022-2024 solo se usa al final, para confirmar las dos alternativas finalistas.

Qué se prueba (cada combinación es una corrida en MLflow, experimento `alerta-insolvencia-iteracion`):
  - Variables: las 19 de la Entrega 2 (base); más 11 de tendencia (cambios frente al año anterior, crecimiento,
    edad, años reportados, pérdidas seguidas); más el grupo NIIF (tamaño).
  - Modelos: random forest (hojas de 30, 50 y 100; max_features 0,3), extra trees, gradient boosting
    regularizado (con y sin balanceo) y regresión logística.
  - Etiqueta ampliada: entrenar con "insolvencia en 1 o 2 años" (más eventos), con purga del año que filtraría
    el futuro.
  - Calibración de Platt de las probabilidades.

Uso (desde la raíz del repositorio, con resumen_empresas disponible):
    export MLFLOW_TRACKING_URI=http://<IP-de-la-EC2>:8050      # o sqlite:///mlflow.db para practicar
    python experiments/iteracion_entrega_final.py --datos data/processed/resumen_empresas.csv
"""

import argparse
import json
import pathlib
import sys
import warnings

import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "model-pkg"))  # reutiliza el recorte, las métricas y la calibración del paquete

from alerta_insolvencia.evaluacion import captura, metricas  # noqa: E402
from alerta_insolvencia.processing.calibracion import ajustar_platt, calibrar  # noqa: E402
from alerta_insolvencia.processing.transformaciones import RecorteExtremos  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

RAZONES = [
    "liquidez_corriente", "endeudamiento", "apalancamiento", "deuda_fin_activos",
    "pasivo_corriente_pasivo", "capital_trabajo_activos", "roa", "roe", "margen_neto",
    "margen_operacional", "cobertura_intereses", "rotacion_activos", "efectivo_activos",
    "ganancias_acum_activos", "ln_activos", "altman_z2",
]
INDICADORES = ["patrimonio_negativo", "perdida_neta"]
TENDENCIA = [
    "edad", "reporto_anio_anterior", "anios_reportados", "d_endeudamiento", "d_roa", "d_liquidez_corriente",
    "d_margen_operacional", "d_efectivo_activos", "crec_activos", "crec_ingresos", "perdidas_seguidas",
]
ANIOS_VALIDACION = [2019, 2020, 2021]
CONJUNTOS = {
    "base": (RAZONES, ["ciiu_grupo"]),
    "tendencia": (RAZONES + TENDENCIA, ["ciiu_grupo"]),
    "tendencia_niif": (RAZONES + TENDENCIA, ["ciiu_grupo", "grupo_niif"]),
}


def bosque(hoja=50, max_features="sqrt"):
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=hoja, max_features=max_features,
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=42)


MODELOS = {
    "random_forest_hoja30": lambda: bosque(30),
    "random_forest_hoja50": lambda: bosque(50),
    "random_forest_hoja100": lambda: bosque(100),
    "random_forest_hoja50_mf0.3": lambda: bosque(50, 0.3),
    "extra_trees_hoja30": lambda: ExtraTreesClassifier(n_estimators=500, min_samples_leaf=30, n_jobs=-1,
                                                       class_weight="balanced_subsample", random_state=42),
    "gradient_boosting_regularizado": lambda: HistGradientBoostingClassifier(
        learning_rate=0.03, max_iter=400, max_leaf_nodes=8, min_samples_leaf=100, l2_regularization=1.0,
        class_weight="balanced", random_state=42),
    "gradient_boosting_sin_balanceo": lambda: HistGradientBoostingClassifier(
        learning_rate=0.03, max_iter=400, max_leaf_nodes=8, min_samples_leaf=100, l2_regularization=1.0,
        random_state=42),
    "logistica_C0.1": lambda: LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000),
}


def cargar(ruta):
    """Lee resumen_empresas y agrega las variables de tendencia (solo con información del pasado de la empresa)."""
    ruta = pathlib.Path(ruta)
    d = pd.read_parquet(ruta) if ruta.suffix == ".parquet" else pd.read_csv(ruta, low_memory=False)
    d["ciiu_grupo"] = d["ciiu_grupo"].astype(str).str[:2]
    d = d.sort_values(["nit", "anio"])
    g = d.groupby("nit")
    anio_previo = g["anio"].shift(1)
    seguido = anio_previo == d["anio"] - 1  # la empresa reportó el año inmediatamente anterior
    d["reporto_anio_anterior"] = seguido.astype(float)
    d["anios_reportados"] = g.cumcount().astype(float)
    d["edad"] = (d["anio"] - pd.to_datetime(d["fecha_constitucion"], errors="coerce").dt.year).clip(lower=0)
    for v in ["endeudamiento", "roa", "liquidez_corriente", "margen_operacional", "efectivo_activos"]:
        d["d_" + v] = (d[v] - g[v].shift(1)).where(seguido)
    d["crec_activos"] = (d["ln_activos"] - g["ln_activos"].shift(1)).where(seguido)
    ingresos = np.log1p(d["ingresos"].clip(lower=0))
    d["crec_ingresos"] = (ingresos - ingresos.groupby(d["nit"]).shift(1)).where(seguido)
    d["perdidas_seguidas"] = ((d["perdida_neta"] == 1) & (g["perdida_neta"].shift(1) == 1) & seguido).astype(float)
    entrada = d["punto_entrada"].astype(str)
    d["grupo_niif"] = np.select(
        [entrada.str.contains("Plenas"), entrada.str.contains("Pymes"), entrada.str.contains("Micro|Grupo 3")],
        ["plenas", "pymes", "micro"], "otro")
    # Etiqueta ampliada: insolvencia en 1 o 2 años (solo para entrenar; siempre se evalúa con y_h1)
    d["y_h12"] = ((d["y_h1"] == 1) | ((d["h2_valido"] == 1) & (d["y_h2"] == 1))).astype(int)
    d = d[d["ciiu_grupo"].isin(["41", "43", "68"]) & (d["post_evento"] == 0) & (d["h1_valido"] == 1)]
    return d.reset_index(drop=True)


def pipeline(numericas, categoricas, modelo, escalar):
    pasos = [("recorte", RecorteExtremos()), ("imputar", SimpleImputer(strategy="median"))]
    if escalar:
        pasos.append(("escalar", StandardScaler()))
    pre = ColumnTransformer([
        ("razones", Pipeline(pasos), numericas),
        ("indicadores", SimpleImputer(strategy="most_frequent"), INDICADORES),
        ("categoricas", OneHotEncoder(handle_unknown="ignore"), categoricas),
    ])
    return Pipeline([("preprocesar", pre), ("modelo", modelo)])


def etiqueta(ent, objetivo, primer_anio_predicho):
    """Con la etiqueta ampliada, el evento a 2 años de las filas recientes cae en un año que todavía no se conoce
    al predecir; esas filas usan solo y_h1 (purga). Sin la purga la validación sube artificialmente."""
    if objetivo == "y_h1":
        return ent["y_h1"]
    return np.where(ent["anio"] <= primer_anio_predicho - 2, ent["y_h12"], ent["y_h1"])


def validacion_movil(d, numericas, categoricas, fabrica, escalar=False, objetivo="y_h1"):
    columnas = numericas + INDICADORES + categoricas
    y, p, a = [], [], []
    for anio in ANIOS_VALIDACION:
        ent, val = d[d["anio"] < anio], d[d["anio"] == anio]
        m = pipeline(numericas, categoricas, fabrica(), escalar).fit(ent[columnas], etiqueta(ent, objetivo, anio))
        y.append(val["y_h1"].to_numpy())
        p.append(m.predict_proba(val[columnas])[:, 1])
        a.append(val["anio"].to_numpy())
    return np.concatenate(y), np.concatenate(p), np.concatenate(a)


def prueba(d, numericas, categoricas, fabrica, objetivo="y_h1"):
    columnas = numericas + INDICADORES + categoricas
    ent, pru = d[d["anio"].between(2016, 2021)], d[d["anio"].between(2022, 2024)]
    m = pipeline(numericas, categoricas, fabrica(), False).fit(ent[columnas], etiqueta(ent, objetivo, 2022))
    return pru["y_h1"].to_numpy(), m.predict_proba(pru[columnas])[:, 1], pru["anio"].to_numpy()


def diferencia_bootstrap(y, p_a, p_b, repeticiones=1000):
    """Diferencia B - A con intervalo del 95 % por remuestreo pareado (las mismas empresas para ambos)."""
    rng = np.random.default_rng(0)
    dif = {"auc": [], "auc_pr": [], "captura_top10": []}
    for _ in range(repeticiones):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() < 2:
            continue
        dif["auc"].append(roc_auc_score(y[i], p_b[i]) - roc_auc_score(y[i], p_a[i]))
        dif["auc_pr"].append(average_precision_score(y[i], p_b[i]) - average_precision_score(y[i], p_a[i]))
        dif["captura_top10"].append(captura(y[i], p_b[i]) - captura(y[i], p_a[i]))
    return {k: {"media": float(np.mean(v)), "ic95": [float(q) for q in np.quantile(v, [0.025, 0.975])]}
            for k, v in dif.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datos", default=str(RAIZ / "data" / "processed" / "resumen_empresas.csv"))
    parser.add_argument("--experimento", default="alerta-insolvencia-iteracion")
    parser.add_argument("--salida", default=str(RAIZ / "experiments" / "resultados_iteracion.json"))
    args = parser.parse_args()

    d = cargar(args.datos)
    experimento = mlflow.set_experiment(args.experimento)
    comunes = {"validacion": "móvil 2019, 2020, 2021 (entrena con los años anteriores)", "prueba": "2022-2024"}
    tabla = []

    # 1. Variables x modelos con la etiqueta de la Entrega 2 (y_h1)
    for conjunto, (num, cat) in CONJUNTOS.items():
        for nombre, fabrica in MODELOS.items():
            y, p, a = validacion_movil(d, num, cat, fabrica, escalar=nombre.startswith("logistica"))
            m = metricas(y, p, a)
            with mlflow.start_run(experiment_id=experimento.experiment_id, run_name=f"{conjunto}__{nombre}"):
                mlflow.log_params({"conjunto_variables": conjunto, "n_variables": len(num) + len(INDICADORES)
                                   + len(cat), "modelo": nombre, "objetivo_entrenamiento": "y_h1", **comunes})
                mlflow.log_metrics({f"validacion_{k}": v for k, v in m.items()})
            tabla.append({"corrida": f"{conjunto}__{nombre}", **{k: m[k] for k in
                          ("auc", "auc_pr", "captura_top10", "captura_top20")}})
            print(f"{conjunto:15} {nombre:32} AUC {m['auc']:.3f}  AUC-PR {m['auc_pr']:.3f}  "
                  f"captura10 {m['captura_top10']:.1%}", flush=True)

    # 2. Etiqueta ampliada (1 o 2 años, con purga) frente a la de la Entrega 2, en validación y en prueba
    finalistas = {}
    for objetivo in ("y_h1", "y_h12"):
        val = validacion_movil(d, RAZONES, ["ciiu_grupo"], MODELOS["random_forest_hoja50"], objetivo=objetivo)
        pru = prueba(d, RAZONES, ["ciiu_grupo"], MODELOS["random_forest_hoja50"], objetivo=objetivo)
        mv, mp = metricas(*val), metricas(*pru)
        finalistas[objetivo] = {"validacion": val, "prueba": pru}
        with mlflow.start_run(experiment_id=experimento.experiment_id,
                              run_name=f"etiqueta_{objetivo}__random_forest_hoja50"):
            mlflow.log_params({"conjunto_variables": "base", "modelo": "random_forest_hoja50",
                               "objetivo_entrenamiento": objetivo, "purga_etiqueta_2_anios": objetivo == "y_h12",
                               **comunes})
            mlflow.log_metrics({**{f"validacion_{k}": v for k, v in mv.items()},
                                **{f"prueba_{k}": v for k, v in mp.items()}})
    comparacion = {
        parte: diferencia_bootstrap(finalistas["y_h1"][parte][0], finalistas["y_h1"][parte][1],
                                    finalistas["y_h12"][parte][1])
        for parte in ("validacion", "prueba")
    }

    # 3. Calibración de Platt del modelo elegido (ajustada en validación, medida en prueba)
    y_val, p_val, _ = finalistas["y_h1"]["validacion"]
    y_pru, p_pru, _ = finalistas["y_h1"]["prueba"]
    calib = ajustar_platt(p_val, y_val)
    p_cal = calibrar(p_pru, **calib)
    calibracion = {
        "coeficientes": calib,
        "brier_sin_calibrar": float(brier_score_loss(y_pru, p_pru)),
        "brier_calibrado": float(brier_score_loss(y_pru, p_cal)),
        "brier_tasa_constante": float(brier_score_loss(y_pru, np.full(len(y_pru), y_pru.mean()))),
        "eventos_observados": int(y_pru.sum()),
        "eventos_esperados_sin_calibrar": float(p_pru.sum()),
        "eventos_esperados_calibrado": float(p_cal.sum()),
    }
    with mlflow.start_run(experiment_id=experimento.experiment_id, run_name="calibracion_platt__random_forest_hoja50"):
        mlflow.log_params({"modelo": "random_forest_hoja50", "calibracion": "Platt",
                           "ajustada_con": "predicciones fuera de tiempo 2019-2021", **comunes})
        mlflow.log_metrics({k: v for k, v in calibracion.items() if isinstance(v, (int, float))})

    resultado = {"validacion": sorted(tabla, key=lambda r: -r["auc_pr"]),
                 "etiqueta_ampliada_menos_entrega2": comparacion, "calibracion": calibracion}
    pathlib.Path(args.salida).write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nEtiqueta ampliada - Entrega 2 (validación):", json.dumps(comparacion["validacion"], indent=1))
    print("Etiqueta ampliada - Entrega 2 (prueba):", json.dumps(comparacion["prueba"], indent=1))
    print("Calibración:", json.dumps(calibracion, indent=1))
    print("Resultados guardados en", args.salida)


if __name__ == "__main__":
    main()
