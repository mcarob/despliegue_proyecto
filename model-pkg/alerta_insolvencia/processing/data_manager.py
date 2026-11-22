"""Lectura de datos, partición temporal y guardado/carga del modelo entrenado y sus metadatos."""

import json
import typing as t
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from alerta_insolvencia import __version__ as _version
from alerta_insolvencia.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config, ruta_datos_entrenamiento

MC = config.modelo


def leer_tabla(ruta: Path) -> pd.DataFrame:
    """Lee resumen_empresas en .csv o .parquet."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta}. Desde la raíz del repositorio ejecute `dvc pull`, o indique la ruta "
            "del archivo con la variable de entorno ALERTA_DATOS."
        )
    if ruta.suffix == ".parquet":
        return pd.read_parquet(ruta)
    return pd.read_csv(ruta, low_memory=False)


def preparar_universo(df: pd.DataFrame) -> pd.DataFrame:
    """Deja el universo del proyecto y las filas que se pueden usar sin fuga de información.

    - Subsectores CIIU 41, 43 y 68 (las obras civiles, 42, quedan por fuera desde la Entrega 1).
    - Sin los años posteriores a la apertura de la insolvencia (post_evento = 1): ya conocen el desenlace.
    """
    df = df.copy()
    df["ciiu_grupo"] = df["ciiu_grupo"].astype(str).str[:2]
    df = df[df["ciiu_grupo"].isin(MC.grupos_ciiu) & (df["post_evento"] == 0)]
    return df.reset_index(drop=True)


def con_etiqueta(df: pd.DataFrame) -> pd.DataFrame:
    """Filas con la respuesta conocida: el año siguiente está dentro del periodo con datos de insolvencia."""
    return df[df["h1_valido"] == 1]


def load_dataset(*, file_name: t.Optional[str] = None) -> pd.DataFrame:
    """Sin argumentos lee la tabla completa de entrenamiento; con file_name, un archivo de datasets/."""
    if file_name:
        return pd.read_csv(DATASET_DIR / file_name, dtype={"ciiu_grupo": str})
    return preparar_universo(leer_tabla(ruta_datos_entrenamiento()))


def particion_temporal(df: pd.DataFrame) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    """Entrenamiento con los años antiguos y prueba con los recientes: el modelo nunca ve los años que predice."""
    etiquetado = con_etiqueta(df)
    entrenamiento = etiquetado[etiquetado["anio"].between(MC.anio_inicio_entrenamiento, MC.anio_fin_entrenamiento)]
    prueba = etiquetado[etiquetado["anio"].between(MC.anio_inicio_prueba, MC.anio_fin_prueba)]
    return entrenamiento, prueba


def nombre_modelo() -> str:
    return f"{config.app_config.archivo_modelo}{_version}.joblib"


def nombre_metadatos() -> str:
    return f"{config.app_config.archivo_metadatos}{_version}.json"


def save_pipeline(*, pipeline_to_persist: Pipeline, metadatos: dict) -> None:
    """Guarda el modelo y sus metadatos con la versión del paquete en el nombre.

    Borra los modelos de versiones anteriores para que el paquete publicado traiga un solo modelo:
    versión del paquete = versión del modelo.
    """
    remove_old_pipelines(files_to_keep=[nombre_modelo(), nombre_metadatos()])
    joblib.dump(pipeline_to_persist, TRAINED_MODEL_DIR / nombre_modelo(), compress=3)
    (TRAINED_MODEL_DIR / nombre_metadatos()).write_text(
        json.dumps(metadatos, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_pipeline(*, file_name: t.Optional[str] = None) -> Pipeline:
    """Carga el modelo entrenado que trae el paquete."""
    ruta = TRAINED_MODEL_DIR / (file_name or nombre_modelo())
    if not ruta.exists():
        raise FileNotFoundError(f"No hay modelo entrenado en {ruta}. Ejecute `tox -e train` primero.")
    return joblib.load(ruta)


def load_metadata() -> dict:
    """Carga los metadatos del modelo: métricas, calibración, umbrales de alerta e importancia de variables."""
    return json.loads((TRAINED_MODEL_DIR / nombre_metadatos()).read_text(encoding="utf-8"))


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """Elimina modelos y metadatos de versiones anteriores."""
    conservar = set(files_to_keep) | {"__init__.py"}
    for archivo in TRAINED_MODEL_DIR.iterdir():
        if archivo.is_file() and archivo.name not in conservar:
            archivo.unlink()
