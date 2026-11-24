"""Segunda ronda de la iteración: búsqueda amplia de un modelo mejor que el random forest de la Entrega 2.

La primera ronda (`iteracion_entrega_final.py`) comparó 8 modelos con configuraciones fijas. Esta ronda busca más a
fondo, con las mismas reglas, fijadas antes de correr:

  - Se decide solo con la validación temporal móvil (2019, 2020 y 2021; se entrena con los años anteriores).
  - Métricas de decisión: AUC-PR y captura en el 10 % con el corte dentro de cada año (como revisaría el banco).
  - Una alternativa reemplaza al modelo actual solo si mejora de forma robusta: el intervalo del 95 % de la diferencia
    (remuestreo pareado dentro de cada año) queda por encima de cero en al menos una métrica de decisión, y la otra
    no empeora. Como se prueban muchas alternativas, la mejor siempre parece algo mejor por azar; por eso se mide
    también cuánto cambian las métricas con solo cambiar la semilla del bosque (el "ruido de fondo").
  - La prueba 2022-2024 se mira una sola vez, al final, para el ganador y el modelo actual.

Qué se prueba (cada alternativa es una corrida en MLflow, experimento `alerta-insolvencia-ronda2`):
  1. Ruido de fondo: el modelo actual con 5 semillas distintas.
  2. Random forest: 26 combinaciones de tamaño de hoja, variables por división, profundidad, balanceo y submuestreo.
  3. Gradient boosting (HistGradientBoosting): 12 combinaciones de tasa de aprendizaje, hojas y regularización.
  4. Submuestreo balanceado (bolsas con todos los eventos y una muestra de no eventos), con árboles y con boosting.
  5. Pocas variables de historia, elegidas por su sentido (pérdidas seguidas, reportó el año anterior, edad,
     crecimiento de ingresos). Cambiarían la entrada de la API, así que solo valen si la ganancia es clara.
  6. Combinaciones (promedio de rangos dentro de cada año) de los mejores de cada familia.
  7. Estabilidad: el actual y el de hojas de 20 con 2.000 árboles y tres semillas.

Uso (desde la raíz del repositorio):
    export MLFLOW_TRACKING_URI=http://<IP-de-la-EC2>:8050      # o sqlite:///mlflow.db para practicar
    python experiments/iteracion_ronda2.py --datos data/processed/resumen_empresas.csv
"""

import argparse
import itertools
import json
import pathlib
import sys
import time

import mlflow
import numpy as np
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

CARPETA = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA))  # reutiliza la carga de datos, el pipeline y la validación de la primera ronda

from iteracion_entrega_final import (  # noqa: E402
    RAIZ, RAZONES, cargar, prueba, validacion_movil,
)
from alerta_insolvencia.evaluacion import captura_por_anio, metricas  # noqa: E402

HISTORIA = ["perdidas_seguidas", "reporto_anio_anterior", "edad", "crec_ingresos"]


class BolsasSubmuestreo(BaseEstimator, ClassifierMixin):
    """Promedio de modelos entrenados con todos los eventos y una muestra distinta de no eventos en cada bolsa.

    Es otra forma de tratar la clase rara: en vez de dar más peso a los eventos, cada modelo ve una muestra
    balanceada (por ejemplo, 20 no eventos por cada evento) y el promedio recupera la información de todos.
    """

    def __init__(self, base=None, n_bolsas=30, no_eventos_por_evento=20, random_state=42):
        self.base = base
        self.n_bolsas = n_bolsas
        self.no_eventos_por_evento = no_eventos_por_evento
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        n_neg = min(len(neg), self.no_eventos_por_evento * len(pos))
        self.modelos_ = []
        for _ in range(self.n_bolsas):
            idx = np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])
            modelo = clone(self.base)
            if "random_state" in modelo.get_params():
                modelo.set_params(random_state=int(rng.integers(1_000_000)))
            self.modelos_.append(modelo.fit(X[idx], y[idx]))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.modelos_], axis=0)
        return np.column_stack([1 - p, p])


def bosque(hoja=50, mf="sqrt", prof=None, peso="balanced_subsample", muestra=None, semilla=42):
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=hoja, max_features=mf, max_depth=prof,
                                  class_weight=peso, max_samples=muestra, n_jobs=-1, random_state=semilla)


def boosting(lr=0.03, hojas=8, min_hoja=100, l2=1.0, iters=400, peso="balanced"):
    return HistGradientBoostingClassifier(learning_rate=lr, max_leaf_nodes=hojas, min_samples_leaf=min_hoja,
                                          l2_regularization=l2, max_iter=iters, class_weight=peso,
                                          random_state=42)


def candidatos():
    """Lista fija de alternativas: (familia, nombre, fábrica del modelo, variables numéricas, parámetros)."""
    lista = []
    # 1. Ruido de fondo: mismo modelo, distinta semilla
    for s in [1, 2, 3, 4, 5]:
        lista.append(("semilla", f"rf_hoja50_semilla{s}", lambda s=s: bosque(semilla=s), RAZONES, {"semilla": s}))
    # 2. Random forest: rejilla fija de 26 combinaciones (sin azar en la elección de la rejilla)
    rejilla_rf = [
        dict(hoja=h, mf=mf, prof=None, peso="balanced_subsample", muestra=None)
        for h, mf in itertools.product([10, 20, 30, 75, 100], ["sqrt", 0.5])
    ] + [
        dict(hoja=50, mf=mf, prof=None, peso="balanced_subsample", muestra=None) for mf in [0.2, 0.3, 0.5]
    ] + [
        dict(hoja=h, mf="sqrt", prof=p, peso="balanced_subsample", muestra=None)
        for h, p in itertools.product([20, 50], [6, 10])
    ] + [
        dict(hoja=h, mf="sqrt", prof=None, peso=peso, muestra=None)
        for h, peso in itertools.product([20, 50], ["balanced", None])
    ] + [
        dict(hoja=h, mf="sqrt", prof=None, peso="balanced_subsample", muestra=m)
        for h, m in itertools.product([20, 50], [0.3, 0.6])
    ] + [dict(hoja=5, mf="sqrt", prof=None, peso="balanced_subsample", muestra=0.3)]
    for c in rejilla_rf:
        nombre = (f"rf_hoja{c['hoja']}_mf{c['mf']}_prof{c['prof']}_peso{c['peso']}_muestra{c['muestra']}")
        lista.append(("random_forest", nombre, lambda c=c: bosque(**c), RAZONES, c))
    # 3. Gradient boosting: 12 combinaciones
    rejilla_gb = [
        dict(lr=lr, hojas=h, min_hoja=mh, l2=1.0, iters=it, peso="balanced")
        for lr, it in [(0.01, 800), (0.03, 300), (0.1, 100)]
        for h, mh in [(4, 200), (8, 100)]
    ] + [
        dict(lr=0.03, hojas=h, min_hoja=mh, l2=l2, iters=300, peso=None)
        for h, mh, l2 in [(4, 200, 1.0), (8, 100, 1.0), (4, 200, 10.0)]
    ] + [
        dict(lr=0.03, hojas=h, min_hoja=mh, l2=10.0, iters=300, peso="balanced")
        for h, mh in [(4, 200), (8, 100), (16, 300)]
    ]
    for c in rejilla_gb:
        nombre = f"gb_lr{c['lr']}_hojas{c['hojas']}_min{c['min_hoja']}_l2{c['l2']}_it{c['iters']}_peso{c['peso']}"
        lista.append(("gradient_boosting", nombre, lambda c=c: boosting(**c), RAZONES, c))
    # 4. Submuestreo balanceado en bolsas
    bolsas = [
        ("bolsas_arbol_hoja20_x20", DecisionTreeClassifier(min_samples_leaf=20, max_features="sqrt"), 200, 20),
        ("bolsas_arbol_hoja5_x10", DecisionTreeClassifier(min_samples_leaf=5, max_features="sqrt"), 200, 10),
        ("bolsas_gb_x20", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=150, max_leaf_nodes=8,
                                                         min_samples_leaf=20, l2_regularization=1.0), 25, 20),
        ("bolsas_gb_x50", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=150, max_leaf_nodes=8,
                                                         min_samples_leaf=20, l2_regularization=1.0), 25, 50),
    ]
    for nombre, base, n, ratio in bolsas:
        lista.append(("submuestreo", nombre,
                      lambda base=base, n=n, ratio=ratio: BolsasSubmuestreo(base, n, ratio),
                      RAZONES, {"n_bolsas": n, "no_eventos_por_evento": ratio, "base": type(base).__name__}))
    # 5. Pocas variables de historia de la empresa
    for extra in [HISTORIA[:2], HISTORIA]:
        nombre = "rf_hoja50_mas_" + "_".join(extra)
        lista.append(("historia", nombre, lambda: bosque(), RAZONES + extra, {"variables_extra": extra}))
    return lista


def rango_por_anio(p, anios):
    """Convierte el puntaje en su rango dentro de cada año (0 a 1), para poder promediar modelos distintos."""
    r = np.empty(len(p))
    for a in np.unique(anios):
        m = anios == a
        r[m] = rankdata(p[m]) / m.sum()
    return r


def diferencia_pareada(y, anios, p_a, p_b, repeticiones=1000):
    """Diferencia B - A con intervalo del 95 %. El remuestreo se hace dentro de cada año (las mismas empresas para
    los dos modelos), para respetar que la captura se mide año por año."""
    rng = np.random.default_rng(0)
    grupos = [np.flatnonzero(anios == a) for a in np.unique(anios)]
    dif = {"auc": [], "auc_pr": [], "captura_top10_por_anio": []}
    for _ in range(repeticiones):
        i = np.concatenate([rng.choice(g, len(g)) for g in grupos])
        dif["auc"].append(roc_auc_score(y[i], p_b[i]) - roc_auc_score(y[i], p_a[i]))
        dif["auc_pr"].append(average_precision_score(y[i], p_b[i]) - average_precision_score(y[i], p_a[i]))
        dif["captura_top10_por_anio"].append(captura_por_anio(y[i], p_b[i], anios[i])
                                             - captura_por_anio(y[i], p_a[i], anios[i]))
    return {k: {"media": float(np.mean(v)), "ic95": [float(q) for q in np.quantile(v, [0.025, 0.975])]}
            for k, v in dif.items()}


def figura(alternativas, m_ref, ruido, ruta):
    """AUC-PR frente a captura por año de cada alternativa, con la franja que cubre el modelo actual con solo
    cambiar la semilla. Lo que cae dentro o cerca de la franja no se distingue del azar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Colores categóricos en orden fijo y una forma distinta por familia (no depende solo del color)
    estilos = {
        "random_forest": ("#2a78d6", "o", "Random forest"),
        "gradient_boosting": ("#eb6834", "s", "Gradient boosting"),
        "submuestreo": ("#1baf7a", "^", "Submuestreo en bolsas"),
        "historia": ("#eda100", "D", "Con variables de historia"),
        "combinacion": ("#e87ba4", "P", "Combinaciones"),
    }
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.axvspan(ruido["auc_pr"]["min"], ruido["auc_pr"]["max"], color="#d9d8d3", alpha=0.6, lw=0,
               label="Modelo actual con otras semillas")
    ax.axhspan(ruido["captura_top10_por_anio"]["min"] * 100, ruido["captura_top10_por_anio"]["max"] * 100,
               color="#d9d8d3", alpha=0.6, lw=0)
    for familia, (color, marca, etiqueta) in estilos.items():
        puntos = [f for f in alternativas if f["familia"] == familia]
        ax.scatter([f["auc_pr"] for f in puntos], [f["captura_top10_por_anio"] * 100 for f in puntos], s=42,
                   color=color, marker=marca, edgecolor="#fcfcfb", linewidth=1,
                   label=f"{etiqueta} ({len(puntos)})", zorder=3)
    ax.scatter([m_ref["auc_pr"]], [m_ref["captura_top10_por_anio"] * 100], s=160, marker="*", color="#0b0b0b",
               zorder=4, label="Modelo actual (Entrega 2)")
    ax.set_xlabel("AUC-PR en validación (más es mejor)")
    ax.set_ylabel("Eventos capturados al revisar el 10 % de cada año (%)")
    ax.set_title(f"Segunda ronda: {len(alternativas)} alternativas frente al modelo actual "
                 "(validación 2019-2021, 94 eventos)",
                 fontsize=10, loc="left")
    ax.grid(color="#e8e7e3", lw=0.6)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(ruta, facecolor="#fcfcfb")
    plt.close(fig)


def resumen(m):
    return {k: m[k] for k in ("auc", "auc_pr", "captura_top10", "captura_top10_por_anio", "captura_top20")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datos", default=str(RAIZ / "data" / "processed" / "resumen_empresas.csv"))
    parser.add_argument("--experimento", default="alerta-insolvencia-ronda2")
    parser.add_argument("--salida", default=str(CARPETA / "resultados_ronda2.json"))
    parser.add_argument("--predicciones", help="archivo .npz opcional para guardar las predicciones de validación")
    parser.add_argument("--figura", default=str(RAIZ / "docs" / "figuras" / "modelos_06_ronda2.png"))
    args = parser.parse_args()

    d = cargar(args.datos)
    experimento = mlflow.set_experiment(args.experimento)
    comunes = {"validacion": "móvil 2019, 2020, 2021 (entrena con los años anteriores)",
               "objetivo_entrenamiento": "y_h1"}

    def registrar(familia, nombre, parametros, m, n_variables):
        with mlflow.start_run(experiment_id=experimento.experiment_id, run_name=nombre):
            mlflow.log_params({"familia": familia, "n_variables": n_variables, **comunes,
                               **{k: str(v) for k, v in parametros.items()}})
            mlflow.log_metrics({f"validacion_{k}": v for k, v in m.items()})

    # Modelo actual (random forest de la Entrega 2): la referencia de todas las comparaciones
    y, p_ref, anios = validacion_movil(d, RAZONES, ["ciiu_grupo"], lambda: bosque())
    m_ref = metricas(y, p_ref, anios)
    registrar("referencia", "actual_rf_hoja50", {"hoja": 50}, m_ref, 19)
    print(f"{'actual_rf_hoja50':60} AUC-PR {m_ref['auc_pr']:.4f}  captura/año {m_ref['captura_top10_por_anio']:.1%}",
          flush=True)

    filas, predicciones = [], {"actual_rf_hoja50": p_ref}
    fabricas = {"actual_rf_hoja50": (lambda: bosque(), RAZONES)}
    for familia, nombre, fabrica, numericas, parametros in candidatos():
        fabricas[nombre] = (fabrica, numericas)
        inicio = time.time()
        y_c, p, a_c = validacion_movil(d, numericas, ["ciiu_grupo"], fabrica)
        assert np.array_equal(y_c, y) and np.array_equal(a_c, anios)  # mismas filas y mismo orden
        m = metricas(y, p, anios)
        registrar(familia, nombre, parametros, m, len(numericas) + 3)
        predicciones[nombre] = p
        filas.append({"familia": familia, "corrida": nombre, **resumen(m)})
        print(f"{nombre:60} AUC-PR {m['auc_pr']:.4f}  captura/año {m['captura_top10_por_anio']:.1%}  "
              f"({time.time() - inicio:.0f} s)", flush=True)

    # 6. Combinaciones: promedio de rangos del mejor de cada familia (y del actual con el mejor boosting)
    mejores = {}
    for f in filas:
        if f["familia"] in ("random_forest", "gradient_boosting", "submuestreo") and (
                f["familia"] not in mejores or f["auc_pr"] > mejores[f["familia"]]["auc_pr"]):
            mejores[f["familia"]] = f
    combinaciones = {
        "combinacion_actual+mejor_gb": ["actual_rf_hoja50", mejores["gradient_boosting"]["corrida"]],
        "combinacion_actual+mejor_submuestreo": ["actual_rf_hoja50", mejores["submuestreo"]["corrida"]],
        "combinacion_mejor_rf+mejor_gb": [mejores["random_forest"]["corrida"],
                                          mejores["gradient_boosting"]["corrida"]],
        "combinacion_tres_familias": [mejores[k]["corrida"] for k in ("random_forest", "gradient_boosting",
                                                                      "submuestreo")],
    }
    # La logística aporta una vista lineal distinta a la de los árboles
    y_l, p_log, _ = validacion_movil(d, RAZONES, ["ciiu_grupo"],
                                     lambda: LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000),
                                     escalar=True)
    predicciones["logistica_C0.1"] = p_log
    fabricas["logistica_C0.1"] = (lambda: LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000),
                                  RAZONES)
    combinaciones["combinacion_actual+logistica"] = ["actual_rf_hoja50", "logistica_C0.1"]
    for nombre, miembros in combinaciones.items():
        p = np.mean([rango_por_anio(predicciones[k], anios) for k in miembros], axis=0)
        m = metricas(y, p, anios)
        registrar("combinacion", nombre, {"miembros": " + ".join(miembros)}, m, 19)
        predicciones[nombre] = p
        filas.append({"familia": "combinacion", "corrida": nombre, **resumen(m)})
        print(f"{nombre:60} AUC-PR {m['auc_pr']:.4f}  captura/año {m['captura_top10_por_anio']:.1%}", flush=True)

    # Ruido de fondo: cuánto se mueven las métricas con solo cambiar la semilla
    semillas = [f for f in filas if f["familia"] == "semilla"] + [{"corrida": "actual", **resumen(m_ref)}]
    ruido = {k: {"min": min(s[k] for s in semillas), "max": max(s[k] for s in semillas)}
             for k in ("auc_pr", "captura_top10_por_anio")}

    # Estabilidad: con 2.000 árboles el azar del bosque pesa menos. Se repite el actual (hojas de 50) y la
    # alternativa de hojas de 20 con tres semillas, para ver si la diferencia entre ellas se sostiene.
    estabilidad = []
    for hoja in (50, 20):
        for s in (42, 1, 2):
            _, p, _ = validacion_movil(d, RAZONES, ["ciiu_grupo"],
                                       lambda hoja=hoja, s=s: bosque(hoja=hoja, semilla=s).set_params(
                                           n_estimators=2000))
            m = metricas(y, p, anios)
            estabilidad.append({"hoja": hoja, "semilla": s, "arboles": 2000, **resumen(m)})
            registrar("estabilidad", f"rf_hoja{hoja}_2000arboles_semilla{s}",
                      {"hoja": hoja, "semilla": s, "arboles": 2000}, m, 19)
            print(f"estabilidad hoja {hoja} semilla {s}: AUC-PR {m['auc_pr']:.4f}  "
                  f"captura/año {m['captura_top10_por_anio']:.1%}", flush=True)

    # Ganador en validación: el de mayor AUC-PR entre las alternativas (sin contar las semillas)
    alternativas = sorted([f for f in filas if f["familia"] != "semilla"], key=lambda f: -f["auc_pr"])
    ganador = alternativas[0]
    comparacion = diferencia_pareada(y, anios, p_ref, predicciones[ganador["corrida"]])
    robusto = any(comparacion[k]["ic95"][0] > 0 for k in ("auc_pr", "captura_top10_por_anio")) and all(
        comparacion[k]["media"] >= 0 for k in ("auc_pr", "captura_top10_por_anio"))
    # Las cinco mejores también se comparan, para ver si alguna otra pasa la regla
    top5 = {f["corrida"]: diferencia_pareada(y, anios, p_ref, predicciones[f["corrida"]])
            for f in alternativas[:5]}

    # Confirmación en la prueba 2022-2024: solo el ganador de validación y el modelo actual, una sola vez
    def puntaje_prueba(nombre):
        miembros = combinaciones.get(nombre, [nombre])
        salidas = []
        for k in miembros:
            fabrica, numericas = fabricas[k]
            y_p, p, a_p = prueba(d, numericas, ["ciiu_grupo"], fabrica, escalar=k.startswith("logistica"))
            salidas.append(p if len(miembros) == 1 else rango_por_anio(p, a_p))
        return y_p, np.mean(salidas, axis=0), a_p

    y_p, p_act, a_p = puntaje_prueba("actual_rf_hoja50")
    _, p_gan, _ = puntaje_prueba(ganador["corrida"])
    confirmacion = {"actual": resumen(metricas(y_p, p_act, a_p)), "ganador": resumen(metricas(y_p, p_gan, a_p)),
                    "ganador_menos_actual": diferencia_pareada(y_p, a_p, p_act, p_gan)}
    print("Prueba 2022-2024 (confirmación):", json.dumps(confirmacion, indent=1))

    resultado = {
        "referencia_actual": resumen(m_ref),
        "ruido_de_fondo_semillas": ruido,
        "ganador_validacion": ganador,
        "ganador_menos_actual_validacion": comparacion,
        "supera_regla_de_decision": bool(robusto),
        "top5_menos_actual_validacion": top5,
        "confirmacion_prueba_2022_2024": confirmacion,
        "miembros_combinaciones": combinaciones,
        "validacion": alternativas,
        "semillas": [f for f in filas if f["familia"] == "semilla"],
        "estabilidad_2000_arboles": estabilidad,
    }
    figura(alternativas, m_ref, ruido, args.figura)
    pathlib.Path(args.salida).write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.predicciones:
        np.savez_compressed(args.predicciones, y=y, anios=anios, **predicciones)
    print("\nRuido de fondo (semillas):", json.dumps(ruido))
    print("Ganador en validación:", ganador["corrida"], json.dumps(comparacion, indent=1))
    print("¿Supera la regla de decisión?", robusto)
    print("Resultados guardados en", args.salida)


if __name__ == "__main__":
    main()
