"""Entrena el modelo, lo evalúa fuera de tiempo, lo calibra y lo guarda dentro del paquete.

Uso (desde model-pkg/, con resumen_empresas disponible; ver README):
    python alerta_insolvencia/train_pipeline.py
o bien `tox -e train`. Si la variable MLFLOW_TRACKING_URI está definida, la corrida se registra en MLflow.

Pasos:
  1. Validación móvil (2019, 2020 y 2021): se entrena con los años anteriores y se predice el año siguiente.
     Esas predicciones fuera de tiempo sirven para medir el modelo sin tocar la prueba y para calibrarlo.
  2. Modelo final con 2016-2021 y evaluación en la prueba 2022-2024, con intervalos de confianza.
  3. Umbrales de alerta con la distribución del puntaje del año más reciente (sin usar su etiqueta).
  4. Se guardan el modelo y los metadatos (métricas, calibración, umbrales, importancia de variables).
"""

import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

# Permite correr el script directamente (python alerta_insolvencia/train_pipeline.py) desde model-pkg/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alerta_insolvencia import __version__ as _version  # noqa: E402
from alerta_insolvencia.config.core import config  # noqa: E402
from alerta_insolvencia.evaluacion import captura, curva_captura, intervalo_bootstrap, metricas  # noqa: E402
from alerta_insolvencia.pipeline import construir_pipeline  # noqa: E402
from alerta_insolvencia.processing.calibracion import ajustar_platt, calibrar  # noqa: E402
from alerta_insolvencia.processing.data_manager import (  # noqa: E402
    con_etiqueta,
    load_dataset,
    particion_temporal,
    save_pipeline,
)

MC = config.modelo


def validacion_movil(datos: pd.DataFrame):
    """Predicciones fuera de tiempo: para cada año de validación, el modelo solo conoce los años anteriores."""
    etiquetado = con_etiqueta(datos)
    y, puntaje, anios = [], [], []
    for anio in MC.anios_validacion:
        ent = etiquetado[etiquetado["anio"].between(MC.anio_inicio_entrenamiento, anio - 1)]
        val = etiquetado[etiquetado["anio"] == anio]
        modelo = construir_pipeline().fit(ent[MC.variables], ent[MC.objetivo])
        y.append(val[MC.objetivo].to_numpy())
        puntaje.append(modelo.predict_proba(val[MC.variables])[:, 1])
        anios.append(val["anio"].to_numpy())
    return np.concatenate(y), np.concatenate(puntaje), np.concatenate(anios)


def calibracion_cruzada(y: np.ndarray, puntaje: np.ndarray, anios: np.ndarray) -> np.ndarray:
    """Probabilidades de validación calibradas sin mirar su propio año: la calibración de cada año se ajusta
    con los otros años de validación. Así el Brier de validación no sale optimista por evaluar la calibración
    con los mismos datos con que se ajustó."""
    probabilidad = np.empty(len(y))
    for anio in np.unique(anios):
        otros = anios != anio
        coef = ajustar_platt(puntaje[otros], y[otros])
        probabilidad[~otros] = calibrar(puntaje[~otros], **coef)
    return probabilidad


def tabla_calibracion(y: np.ndarray, probabilidad: np.ndarray, grupos: int = 10) -> list:
    """Eventos observados frente a esperados por decil de riesgo (decil 10 = mayor riesgo)."""
    decil = pd.qcut(pd.Series(probabilidad).rank(method="first"), grupos, labels=False) + 1
    tabla = pd.DataFrame({"decil": decil, "y": y, "p": probabilidad}).groupby("decil").agg(
        empresas=("y", "size"), eventos_observados=("y", "sum"), eventos_esperados=("p", "sum"))
    tabla["eventos_esperados"] = tabla["eventos_esperados"].round(2)
    return tabla.reset_index().astype({"decil": int, "empresas": int, "eventos_observados": int}).to_dict("records")


def importancia(modelo) -> dict:
    """Importancia de cada variable en el random forest (reducción media de impureza)."""
    categorias = modelo.named_steps["preprocesar"].named_transformers_["ciiu"].categories_[0]
    nombres = MC.razones + MC.indicadores + [f"ciiu_{c}" for c in categorias]
    valores = modelo.named_steps["modelo"].feature_importances_
    return dict(sorted(zip(nombres, map(float, valores)), key=lambda kv: -kv[1]))


def registrar_en_mlflow(modelo, metadatos: dict, datos_prueba: pd.DataFrame) -> None:
    """Registra la corrida en MLflow si hay servidor configurado (MLFLOW_TRACKING_URI)."""
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        print("MLFLOW_TRACKING_URI no está definida: la corrida no se registra en MLflow.")
        return
    import mlflow
    import mlflow.sklearn

    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "alerta-insolvencia"))
    with mlflow.start_run(run_name=f"paquete_v{_version}_rf_calibrado"):
        mlflow.log_params({
            "modelo": "RandomForestClassifier + calibración de Platt",
            "version_paquete": _version,
            "n_estimators": MC.n_estimators, "min_samples_leaf": MC.min_samples_leaf,
            "class_weight": MC.class_weight, "random_state": MC.random_state,
            "anios_entrenamiento": metadatos["datos"]["anios_entrenamiento"],
            "anios_validacion": ",".join(map(str, MC.anios_validacion)),
            "anios_prueba": metadatos["datos"]["anios_prueba"],
            "variables": len(MC.variables),
        })
        mlflow.log_metrics({f"validacion_{k}": v for k, v in metadatos["metricas_validacion"].items()})
        mlflow.log_metrics({f"prueba_{k}": v for k, v in metadatos["metricas_prueba"].items()})
        mlflow.log_metric("prueba_brier_sin_calibrar", metadatos["brier_sin_calibrar"])
        mlflow.log_dict(metadatos, "metadatos_modelo.json")
        ejemplo = datos_prueba[MC.variables].head(5)
        mlflow.sklearn.log_model(modelo, name="modelo", serialization_format="cloudpickle", input_example=ejemplo)
    print("Corrida registrada en MLflow:", os.environ["MLFLOW_TRACKING_URI"])


def run_training() -> dict:
    datos = load_dataset()
    entrenamiento, prueba = particion_temporal(datos)
    print(f"Entrenamiento {MC.anio_inicio_entrenamiento}-{MC.anio_fin_entrenamiento}: {len(entrenamiento)} "
          f"empresa-año, {int(entrenamiento[MC.objetivo].sum())} eventos | Prueba {MC.anio_inicio_prueba}-"
          f"{MC.anio_fin_prueba}: {len(prueba)} empresa-año, {int(prueba[MC.objetivo].sum())} eventos")

    # 1. Validación móvil y calibración con predicciones fuera de tiempo
    y_val, p_val, anios_val = validacion_movil(datos)
    calib = ajustar_platt(p_val, y_val)
    p_val_cal = calibracion_cruzada(y_val, p_val, anios_val)
    metricas_validacion = metricas(y_val, p_val, anios_val, p_val_cal)
    # Tasa real frente a probabilidad media por año: muestra cuánto varía la tasa del sector de un año a otro
    tasa_por_anio = {int(a): {"tasa_real": float(y_val[anios_val == a].mean()),
                              "probabilidad_media": float(p_val_cal[anios_val == a].mean())}
                     for a in np.unique(anios_val)}

    # 2. Modelo final y prueba fuera de tiempo
    modelo = construir_pipeline().fit(entrenamiento[MC.variables], entrenamiento[MC.objetivo])
    y = prueba[MC.objetivo].to_numpy()
    puntaje = modelo.predict_proba(prueba[MC.variables])[:, 1]
    probabilidad = calibrar(puntaje, **calib)
    metricas_prueba = metricas(y, puntaje, prueba["anio"].to_numpy(), probabilidad)

    intervalos = {
        "auc": intervalo_bootstrap(y, puntaje, roc_auc_score),
        "auc_pr": intervalo_bootstrap(y, puntaje, average_precision_score),
        "captura_top10": intervalo_bootstrap(y, puntaje, lambda a, b: captura(a, b, 0.10)),
    }
    # Líneas base sin entrenamiento: a menor Z'' de Altman o mayor endeudamiento, más riesgo
    altman = -prueba["altman_z2"].fillna(prueba["altman_z2"].median()).to_numpy()
    endeud = prueba["endeudamiento"].fillna(prueba["endeudamiento"].median()).to_numpy()

    # 3. Umbrales de alerta con el año más reciente (sus puntajes; su etiqueta todavía no existe)
    anio_ref = int(datos["anio"].max())
    p_ref = modelo.predict_proba(datos.loc[datos["anio"] == anio_ref, MC.variables])[:, 1]
    umbrales = {
        "anio_referencia": anio_ref,
        "alto": float(np.quantile(p_ref, 1 - MC.fraccion_alerta_alta)),
        "medio": float(np.quantile(p_ref, 1 - MC.fraccion_alerta_alta - MC.fraccion_alerta_media)),
    }

    metadatos = {
        "modelo": "random_forest_hoja50 calibrado (Platt)",
        "version": _version,
        "fecha_entrenamiento": date.today().isoformat(),
        "objetivo": "Apertura de un proceso de insolvencia en el año siguiente al de los estados financieros (y_h1)",
        "variables": {"razones": MC.razones, "indicadores": MC.indicadores, "categoricas": MC.categoricas},
        "datos": {
            "universo": "CIIU " + ", ".join(MC.grupos_ciiu),
            "anios_entrenamiento": f"{MC.anio_inicio_entrenamiento}-{MC.anio_fin_entrenamiento}",
            "anios_validacion": MC.anios_validacion,
            "anios_prueba": f"{MC.anio_inicio_prueba}-{MC.anio_fin_prueba}",
            "filas_entrenamiento": int(len(entrenamiento)),
            "eventos_entrenamiento": int(entrenamiento[MC.objetivo].sum()),
            "filas_prueba": int(len(y)), "eventos_prueba": int(y.sum()),
        },
        "metricas_validacion": metricas_validacion,
        "metricas_prueba": metricas_prueba,
        "intervalos_95_prueba": intervalos,
        "brier_sin_calibrar": float(brier_score_loss(y, puntaje)),
        "brier_tasa_constante": float(brier_score_loss(y, np.full(len(y), y.mean()))),
        "lineas_base_prueba": {
            "altman_z2": metricas(y, altman), "endeudamiento": metricas(y, endeud),
        },
        "calibracion": {**calib, "metodo": "Platt sobre predicciones fuera de tiempo de "
                                           + ", ".join(map(str, MC.anios_validacion))},
        "tabla_calibracion_prueba": tabla_calibracion(y, probabilidad),
        "calibracion_en_conjunto": {
            "validacion_por_anio": tasa_por_anio,
            "prueba": {"eventos_observados": int(y.sum()), "eventos_esperados": float(probabilidad.sum()),
                       "tasa_real": float(y.mean()), "probabilidad_media": float(probabilidad.mean())},
        },
        "umbrales_alerta": umbrales,
        "curva_captura_prueba": dict(zip(("revisadas", "capturados"), curva_captura(y, puntaje))),
        "importancia": importancia(modelo),
    }
    save_pipeline(pipeline_to_persist=modelo, metadatos=metadatos)

    mv, m = metricas_validacion, metricas_prueba
    print(f"Validación móvil: AUC {mv['auc']:.3f}, captura 10 % {mv['captura_top10']:.1%}")
    print(f"Prueba: AUC {m['auc']:.3f} (IC 95 % {intervalos['auc'][0]:.3f}-{intervalos['auc'][1]:.3f}), "
          f"AUC-PR {m['auc_pr']:.3f}, captura 10 % {m['captura_top10']:.1%}, "
          f"Brier {metadatos['brier_sin_calibrar']:.4f} -> {m['brier']:.4f} calibrado")
    print(f"Umbrales de alerta ({anio_ref}): alto >= {umbrales['alto']:.3f}, medio >= {umbrales['medio']:.3f}")
    registrar_en_mlflow(modelo, metadatos, prueba)
    return metadatos


if __name__ == "__main__":
    run_training()
