"""Modelo que usa el tablero: el random forest ganador de la Entrega 2 (random_forest_hoja50).

Se entrena con la misma configuración de experiments/entrenar_modelos.py (mismas variables, preprocesamiento,
hiperparámetros y semilla), así el tablero no depende de que la EC2 con MLflow esté encendida.
En la entrega final esta parte se reemplaza por el paquete del modelo servido por la API (POST /predict):
basta con cambiar la función calcular_riesgo().
"""

import json
import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from datos import CARPETA_TABLERO, cargar_datos
from transformaciones import RecorteExtremos

CARPETA_ARTEFACTOS = CARPETA_TABLERO / "artefactos"
RUTA_MODELO = CARPETA_ARTEFACTOS / "modelo_rf.joblib"
RUTA_PUNTAJES = CARPETA_ARTEFACTOS / "puntajes.csv"
RUTA_METRICAS = CARPETA_ARTEFACTOS / "metricas.json"

NOMBRE_MODELO = "random_forest_hoja50"
VERSION_MODELO = "v1.0"
ANIOS_ENTRENAMIENTO = range(2016, 2022)  # 2016-2021, igual que experiments/entrenar_modelos.py
ANIOS_PRUEBA = range(2022, 2025)  # 2022-2024

RAZONES = [
    "liquidez_corriente", "endeudamiento", "apalancamiento", "deuda_fin_activos",
    "pasivo_corriente_pasivo", "capital_trabajo_activos", "roa", "roe", "margen_neto",
    "margen_operacional", "cobertura_intereses", "rotacion_activos", "efectivo_activos",
    "ganancias_acum_activos", "ln_activos", "altman_z2",
]
INDICADORES = ["patrimonio_negativo", "perdida_neta"]
CATEGORICAS = ["ciiu_grupo"]
VARIABLES = RAZONES + INDICADORES + CATEGORICAS

# Nombres para mostrar en el tablero
ETIQUETAS = {
    "liquidez_corriente": "Liquidez corriente", "endeudamiento": "Endeudamiento (pasivo/activo)",
    "apalancamiento": "Apalancamiento (pasivo/patrimonio)", "deuda_fin_activos": "Deuda financiera / activos",
    "pasivo_corriente_pasivo": "Pasivo corriente / pasivo", "capital_trabajo_activos": "Capital de trabajo / activos",
    "roa": "ROA", "roe": "ROE", "margen_neto": "Margen neto", "margen_operacional": "Margen operacional",
    "cobertura_intereses": "Cobertura de intereses", "rotacion_activos": "Rotación de activos",
    "efectivo_activos": "Efectivo / activos", "ganancias_acum_activos": "Ganancias acumuladas / activos",
    "ln_activos": "Tamaño (log de activos)", "altman_z2": "Z'' de Altman",
    "patrimonio_negativo": "Patrimonio negativo", "perdida_neta": "Pérdida neta",
}


def crear_modelo():
    """Pipeline idéntico a la corrida random_forest_hoja50 registrada en MLflow."""
    preprocesar = ColumnTransformer([
        ("razones", Pipeline([("recorte", RecorteExtremos()), ("imputar", SimpleImputer(strategy="median"))]),
         RAZONES),
        ("indicadores", SimpleImputer(strategy="most_frequent"), INDICADORES),
        ("ciiu", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
    ])
    bosque = RandomForestClassifier(n_estimators=500, min_samples_leaf=50, class_weight="balanced_subsample",
                                    n_jobs=-1, random_state=42)
    return Pipeline([("preprocesar", preprocesar), ("modelo", bosque)])


def captura(y, puntaje, fraccion=0.10):
    corte = np.quantile(puntaje, 1 - fraccion)
    return y[puntaje >= corte].sum() / y.sum()


def curva_captura(y, puntaje, pasos=50):
    orden = np.argsort(-puntaje)
    acumulado = np.cumsum(y[orden]) / y.sum()
    x = np.linspace(0, 1, pasos + 1)
    idx = np.clip((x * len(y)).astype(int) - 1, 0, len(y) - 1)
    return [0.0] + list(x[1:]), [0.0] + list(acumulado[idx[1:]])


def metricas(y, puntaje):
    return {
        "auc": float(roc_auc_score(y, puntaje)),
        "auc_pr": float(average_precision_score(y, puntaje)),
        "captura_top10": float(captura(y, puntaje)),
    }


def psi(referencia, actual, cortes=10):
    """Índice de estabilidad poblacional (deriva) de una variable entre entrenamiento y un año reciente."""
    referencia, actual = referencia.dropna(), actual.dropna()
    bordes = np.unique(np.quantile(referencia, np.linspace(0, 1, cortes + 1)))
    if len(bordes) < 3 or len(actual) == 0:
        return float("nan")
    bordes[0], bordes[-1] = -np.inf, np.inf
    p = np.histogram(referencia, bordes)[0] / len(referencia)
    q = np.histogram(actual, bordes)[0] / len(actual)
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


def preparar(forzar=False):
    """Entrena el modelo, calcula el puntaje de todas las empresas-año y las métricas. Se ejecuta una sola vez."""
    if not forzar and RUTA_MODELO.exists() and RUTA_PUNTAJES.exists() and RUTA_METRICAS.exists():
        return
    CARPETA_ARTEFACTOS.mkdir(exist_ok=True)
    print("Entrenando el modelo del tablero (solo la primera vez, tarda menos de un minuto)...")
    d = cargar_datos()
    modelable = d[(d["post_evento"] == 0)]
    entrenamiento = modelable[modelable["anio"].isin(ANIOS_ENTRENAMIENTO) & (modelable["h1_valido"] == 1)]
    prueba = modelable[modelable["anio"].isin(ANIOS_PRUEBA) & (modelable["h1_valido"] == 1)]

    modelo = crear_modelo()
    modelo.fit(entrenamiento[VARIABLES], entrenamiento["y_h1"].to_numpy())
    joblib.dump(modelo, RUTA_MODELO)

    # Puntaje de todas las empresas-año del universo (incluye 2025, el año más reciente, sin etiqueta todavía)
    puntajes = modelable[["nit", "anio"]].copy()
    puntajes["puntaje"] = modelo.predict_proba(modelable[VARIABLES])[:, 1]
    puntajes.to_csv(RUTA_PUNTAJES, index=False)

    # Métricas en la prueba fuera de tiempo (2022-2024) frente a las líneas base de la Entrega 1
    y = prueba["y_h1"].to_numpy()
    p_modelo = modelo.predict_proba(prueba[VARIABLES])[:, 1]
    p_altman = -prueba["altman_z2"].fillna(prueba["altman_z2"].median()).to_numpy()
    p_endeud = prueba["endeudamiento"].fillna(prueba["endeudamiento"].median()).to_numpy()
    resultado = {
        "modelo": NOMBRE_MODELO, "version": VERSION_MODELO,
        "entrenamiento": "2016-2021", "prueba": "2022-2024",
        "filas_entrenamiento": int(len(entrenamiento)), "eventos_entrenamiento": int(entrenamiento["y_h1"].sum()),
        "filas_prueba": int(len(prueba)), "eventos_prueba": int(y.sum()),
        "prevalencia_prueba": float(y.mean()),
        "comparacion": {
            "Z'' de Altman (línea base)": metricas(y, p_altman),
            "Endeudamiento (línea base)": metricas(y, p_endeud),
            f"Modelo {VERSION_MODELO} (random forest)": metricas(y, p_modelo),
        },
        "brier": float(brier_score_loss(y, p_modelo)),
        "curvas": {
            "Modelo": curva_captura(y, p_modelo),
            "Z'' de Altman": curva_captura(y, p_altman),
        },
        "importancia": dict(zip(
            RAZONES + INDICADORES + [f"ciiu_{c}" for c in
                                     modelo.named_steps["preprocesar"].named_transformers_["ciiu"].categories_[0]],
            map(float, modelo.named_steps["modelo"].feature_importances_))),
    }
    ultimo = int(modelable["anio"].max())
    resultado["psi_anio"] = ultimo
    resultado["psi"] = {v: psi(entrenamiento[v], modelable.loc[modelable["anio"] == ultimo, v])
                        for v in ["endeudamiento", "liquidez_corriente", "efectivo_activos", "deuda_fin_activos",
                                  "roa", "margen_operacional"]}
    RUTA_METRICAS.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    m = resultado["comparacion"][f"Modelo {VERSION_MODELO} (random forest)"]
    print(f"Listo. Prueba 2022-2024: AUC {m['auc']:.3f}, captura en el 10 % {m['captura_top10']:.1%}")


_modelo = None


def calcular_riesgo(registros):
    """Puntaje de riesgo para una o varias empresas (lista de diccionarios con las variables del modelo).

    Hoy usa el modelo cargado en memoria. En la entrega final esta función hará POST /predict a la API
    con el mismo formato de entrada: {"inputs": registros}.
    """
    global _modelo
    if _modelo is None:
        _modelo = joblib.load(RUTA_MODELO)
    X = pd.DataFrame(registros).reindex(columns=VARIABLES)
    X[RAZONES + INDICADORES] = X[RAZONES + INDICADORES].astype(float)
    return _modelo.predict_proba(X)[:, 1]


def explicar(fila, medianas):
    """Aporte aproximado de cada variable: cuánto baja o sube el puntaje si esa variable tomara
    el valor mediano de su subsector (las demás quedan iguales). Positivo = la variable aumenta el riesgo."""
    base = fila[VARIABLES].to_dict()
    variantes = [base]
    for v in RAZONES:
        alterna = dict(base)
        alterna[v] = medianas.get(v, np.nan)
        variantes.append(alterna)
    p = calcular_riesgo(variantes)
    return pd.Series(p[0] - p[1:], index=RAZONES).sort_values(key=abs, ascending=False)


def cargar_metricas():
    return json.loads(RUTA_METRICAS.read_text(encoding="utf-8"))


if __name__ == "__main__":
    preparar(forzar=True)
