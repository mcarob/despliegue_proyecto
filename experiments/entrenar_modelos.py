"""Entrena los modelos de alerta temprana de insolvencia y registra cada corrida en MLflow.

Uso (con el servidor de MLflow corriendo en la EC2; en PowerShell: $env:MLFLOW_TRACKING_URI = "...")
    export MLFLOW_TRACKING_URI=http://<IP-de-la-EC2>:8050
    python experiments/entrenar_modelos.py --datos data/processed/resumen_empresas.csv

Cada corrida registra sus parámetros, las métricas de la prueba fuera de tiempo y el modelo entrenado.
"""

import argparse
import pathlib

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = pathlib.Path(__file__).resolve().parent.parent

# Universo del proyecto: construcción de edificios (41), actividades especializadas (43) e inmobiliarias (68).
# Las obras civiles (42) quedan por fuera, como se definió en la Entrega 1.
GRUPOS_CIIU = ("41", "43", "68")

# Partición temporal: se entrena con los años antiguos y se prueba con los recientes (sin mezclar años).
ANIOS_ENTRENAMIENTO = range(2016, 2022)  # 2016-2021
ANIOS_PRUEBA = range(2022, 2025)  # 2022-2024 (2025 no tiene la etiqueta completa)

OBJETIVO = "y_h1"  # 1 si la empresa abre un proceso de insolvencia al año siguiente

RAZONES = [
    "liquidez_corriente", "endeudamiento", "apalancamiento", "deuda_fin_activos",
    "pasivo_corriente_pasivo", "capital_trabajo_activos", "roa", "roe", "margen_neto",
    "margen_operacional", "cobertura_intereses", "rotacion_activos", "efectivo_activos",
    "ganancias_acum_activos", "ln_activos", "altman_z2",
]
INDICADORES = ["patrimonio_negativo", "perdida_neta"]
CATEGORICAS = ["ciiu_grupo"]


class RecorteExtremos(BaseEstimator, TransformerMixin):
    """Recorta cada razón a sus percentiles 1 y 99 calculados en entrenamiento (las razones tienen colas enormes)."""

    def __init__(self, inferior=0.01, superior=0.99):
        self.inferior = inferior
        self.superior = superior

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.limites_ = (X.quantile(self.inferior), X.quantile(self.superior))
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.clip(self.limites_[0], self.limites_[1], axis=1).to_numpy()


def cargar_datos(ruta):
    """Lee la tabla resumen_empresas y deja solo las filas válidas para modelar."""
    ruta = pathlib.Path(ruta)
    df = pd.read_parquet(ruta) if ruta.suffix == ".parquet" else pd.read_csv(ruta, low_memory=False)
    df["ciiu_grupo"] = df["ciiu_grupo"].astype(str).str[:2]
    df = df[
        df["ciiu_grupo"].isin(GRUPOS_CIIU)
        & (df["post_evento"] == 0)  # los años posteriores a la insolvencia filtran el futuro
        & (df["h1_valido"] == 1)  # el año siguiente debe estar dentro del periodo con etiqueta
    ]
    entrenamiento = df[df["anio"].isin(ANIOS_ENTRENAMIENTO)]
    prueba = df[df["anio"].isin(ANIOS_PRUEBA)]
    return entrenamiento, prueba


def preprocesador(escalar):
    """Recorta extremos, imputa faltantes con la mediana y codifica el subsector CIIU."""
    pasos_razones = [("recorte", RecorteExtremos()), ("imputar", SimpleImputer(strategy="median"))]
    if escalar:  # la regresión logística necesita variables en la misma escala; los árboles no
        pasos_razones.append(("escalar", StandardScaler()))
    return ColumnTransformer([
        ("razones", Pipeline(pasos_razones), RAZONES),
        ("indicadores", SimpleImputer(strategy="most_frequent"), INDICADORES),
        ("ciiu", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
    ])


def configuraciones():
    """Modelos y variantes de hiperparámetros a comparar (cada una es una corrida en MLflow)."""
    return [
        ("logistica_C1", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000), True),
        ("logistica_C0.1", LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000), True),
        ("random_forest_hoja20", RandomForestClassifier(
            n_estimators=500, min_samples_leaf=20, class_weight="balanced_subsample",
            n_jobs=-1, random_state=42), False),
        ("random_forest_hoja50", RandomForestClassifier(
            n_estimators=500, min_samples_leaf=50, class_weight="balanced_subsample",
            n_jobs=-1, random_state=42), False),
        ("gradient_boosting_lr0.05", HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=15, class_weight="balanced",
            random_state=42), False),
        ("gradient_boosting_lr0.1", HistGradientBoostingClassifier(
            learning_rate=0.1, max_iter=200, max_leaf_nodes=15, class_weight="balanced",
            random_state=42), False),
    ]


def metricas(y, puntaje):
    """Métricas para una clase rara: AUC, AUC-PR, Brier y captura de eventos en el 10 % de mayor riesgo."""
    prevalencia = y.mean()
    corte = np.quantile(puntaje, 0.90)
    captura = y[puntaje >= corte].sum() / y.sum()
    auc_pr = average_precision_score(y, puntaje)
    return {
        "auc": roc_auc_score(y, puntaje),
        "auc_pr": auc_pr,
        "auc_pr_sobre_prevalencia": auc_pr / prevalencia,
        "captura_top10": captura,
        "prevalencia_prueba": prevalencia,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", default=str(RAIZ / "data" / "processed" / "resumen_empresas.parquet"))
    parser.add_argument("--experimento", default="alerta-insolvencia")
    args = parser.parse_args()

    entrenamiento, prueba = cargar_datos(args.datos)
    X_ent, y_ent = entrenamiento[RAZONES + INDICADORES + CATEGORICAS], entrenamiento[OBJETIVO].to_numpy()
    X_pru, y_pru = prueba[RAZONES + INDICADORES + CATEGORICAS], prueba[OBJETIVO].to_numpy()
    print(f"Entrenamiento: {len(y_ent)} empresa-año, {y_ent.sum()} eventos | "
          f"Prueba: {len(y_pru)} empresa-año, {y_pru.sum()} eventos")

    experimento = mlflow.set_experiment(args.experimento)
    datos_comunes = {
        "anios_entrenamiento": "2016-2021", "anios_prueba": "2022-2024",
        "ciiu": "41, 43, 68", "objetivo": OBJETIVO,
        "filas_entrenamiento": len(y_ent), "eventos_entrenamiento": int(y_ent.sum()),
        "filas_prueba": len(y_pru), "eventos_prueba": int(y_pru.sum()),
    }
    resultados = []

    # Línea base sin entrenamiento: el Z'' de Altman (menor Z, más riesgo), la referencia de la Entrega 1
    with mlflow.start_run(experiment_id=experimento.experiment_id, run_name="linea_base_altman"):
        mlflow.log_params({"modelo": "Altman Z'' (sin entrenar)", **datos_comunes})
        z = prueba["altman_z2"].fillna(prueba["altman_z2"].median()).to_numpy()
        m = metricas(y_pru, -z)
        mlflow.log_metrics(m)
        resultados.append(("linea_base_altman", m))

    for nombre, modelo, escalar in configuraciones():
        with mlflow.start_run(experiment_id=experimento.experiment_id, run_name=nombre):
            pipeline = Pipeline([("preprocesar", preprocesador(escalar)), ("modelo", modelo)])
            pipeline.fit(X_ent, y_ent)
            probabilidad = pipeline.predict_proba(X_pru)[:, 1]

            m = metricas(y_pru, probabilidad)
            # Con class_weight las probabilidades quedan infladas: el Brier sirve para comparar, no como probabilidad real
            m["brier"] = brier_score_loss(y_pru, probabilidad)
            mlflow.log_params({"modelo": type(modelo).__name__, **modelo.get_params(), **datos_comunes})
            mlflow.log_metrics(m)
            mlflow.sklearn.log_model(pipeline, name="modelo", serialization_format="cloudpickle")
            resultados.append((nombre, m))

    print(f"\n{'Corrida':28} {'AUC':>6} {'AUC-PR':>7} {'xPrev':>6} {'Top10%':>7}")
    for nombre, m in resultados:
        print(f"{nombre:28} {m['auc']:6.3f} {m['auc_pr']:7.3f} "
              f"{m['auc_pr_sobre_prevalencia']:6.1f} {m['captura_top10']:7.1%}")


if __name__ == "__main__":
    main()
