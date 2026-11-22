"""Predicción con el modelo entrenado que trae el paquete."""

import typing as t

import pandas as pd

from alerta_insolvencia import __version__ as _version
from alerta_insolvencia.config.core import config
from alerta_insolvencia.processing.calibracion import calibrar, nivel_alerta
from alerta_insolvencia.processing.data_manager import load_metadata, load_pipeline
from alerta_insolvencia.processing.validation import validate_inputs

# El modelo y sus metadatos se cargan una sola vez, al importar el módulo
_pipeline = load_pipeline()
_metadatos = load_metadata()
_calibracion = _metadatos["calibracion"]
_umbrales = _metadatos["umbrales_alerta"]


def make_prediction(*, input_data: t.Union[pd.DataFrame, dict, list]) -> dict:
    """Puntaje, probabilidad calibrada y nivel de alerta para una o varias empresas.

    Devuelve un diccionario con:
      - predictions: puntaje de riesgo de 0 a 1 de cada empresa (sirve para ordenar; es el que usa el tablero);
      - probabilidades: probabilidad calibrada de abrir insolvencia el año siguiente;
      - niveles_alerta: "alto", "medio" o "bajo" según los umbrales del año de referencia;
      - version: versión del modelo; errors: None o la lista de errores de validación.
    """
    datos = pd.DataFrame(input_data)
    resultados = {"predictions": None, "probabilidades": None, "niveles_alerta": None,
                  "version": _version, "errors": None}

    maximo = config.app_config.max_registros_por_llamada
    if len(datos) == 0 or len(datos) > maximo:
        resultados["errors"] = [{"type": "tamano_lote", "loc": ["inputs"],
                                 "msg": f"Envíe entre 1 y {maximo} registros por llamada (llegaron {len(datos)})."}]
        return resultados

    validados, errores = validate_inputs(input_data=datos)
    if errores:
        resultados["errors"] = errores
        return resultados

    puntaje = _pipeline.predict_proba(validados[config.modelo.variables])[:, 1]
    probabilidad = calibrar(puntaje, _calibracion["coeficiente"], _calibracion["intercepto"])
    resultados.update({
        "predictions": [float(p) for p in puntaje],
        "probabilidades": [float(p) for p in probabilidad],
        "niveles_alerta": nivel_alerta(puntaje, _umbrales["alto"], _umbrales["medio"]),
    })
    return resultados


def model_metadata() -> dict:
    """Metadatos del modelo (métricas, periodos, calibración, umbrales e importancia de variables)."""
    return _metadatos
