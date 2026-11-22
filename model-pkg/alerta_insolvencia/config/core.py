"""Lee y valida config.yml. El resto del paquete toma de aquí rutas, variables y parámetros."""

import os
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel

import alerta_insolvencia

# Carpetas del paquete
PACKAGE_ROOT = Path(alerta_insolvencia.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained"


class AppConfig(BaseModel):
    """Nombres de archivos y rutas."""

    nombre_paquete: str
    archivo_modelo: str
    archivo_metadatos: str
    datos_entrenamiento: str
    datos_prueba: str
    max_registros_por_llamada: int


class ModelConfig(BaseModel):
    """Todo lo que define el entrenamiento y las variables del modelo."""

    objetivo: str
    grupos_ciiu: List[str]
    razones: List[str]
    indicadores: List[str]
    categoricas: List[str]
    anio_inicio_entrenamiento: int
    anio_fin_entrenamiento: int
    anios_validacion: List[int]
    anio_inicio_prueba: int
    anio_fin_prueba: int
    n_estimators: int
    min_samples_leaf: int
    class_weight: str
    random_state: int
    recorte_inferior: float
    recorte_superior: float
    fraccion_alerta_alta: float
    fraccion_alerta_media: float

    @property
    def variables(self) -> List[str]:
        """Columnas que entran al modelo, en el orden en que las espera el pipeline."""
        return self.razones + self.indicadores + self.categoricas


class Config(BaseModel):
    """Configuración completa."""

    app_config: AppConfig
    modelo: ModelConfig


def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> dict:
    """Lee el archivo YAML de configuración."""
    cfg_path = cfg_path or CONFIG_FILE_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(f"No se encontró la configuración en {cfg_path}")
    with open(cfg_path, encoding="utf-8") as archivo:
        return yaml.safe_load(archivo)


def create_and_validate_config(parsed_config: Optional[dict] = None) -> Config:
    """Valida los tipos de cada valor (un error en config.yml se detecta al importar, no al entrenar)."""
    parsed_config = parsed_config or fetch_config_from_yaml()
    return Config(app_config=AppConfig(**parsed_config), modelo=ModelConfig(**parsed_config))


def ruta_datos_entrenamiento() -> Path:
    """Ruta de resumen_empresas: la variable de entorno ALERTA_DATOS tiene prioridad sobre config.yml."""
    if os.environ.get("ALERTA_DATOS"):
        return Path(os.environ["ALERTA_DATOS"])
    return (ROOT / config.app_config.datos_entrenamiento).resolve()


config = create_and_validate_config()
