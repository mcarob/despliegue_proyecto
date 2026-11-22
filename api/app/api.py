"""Rutas de la API: estado (/health), predicción (/predict) y ficha del modelo (/modelo)."""

from typing import Any

import pandas as pd
from alerta_insolvencia import __version__ as model_version
from alerta_insolvencia.predict import make_prediction, model_metadata
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from loguru import logger

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """Confirma que la API está en línea y dice qué versión del modelo sirve."""
    estado = schemas.Health(name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version)
    return estado.model_dump()


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
def predict(input_data: schemas.MultipleDataInputs) -> Any:
    """Puntaje de riesgo, probabilidad calibrada y nivel de alerta de cada empresa enviada."""
    datos = pd.DataFrame(jsonable_encoder(input_data.inputs))
    logger.info(f"Predicción solicitada para {len(datos)} empresa(s)")
    resultados = make_prediction(input_data=datos)

    if resultados["errors"] is not None:
        logger.warning(f"Error de validación: {resultados['errors']}")
        raise HTTPException(status_code=400, detail=resultados["errors"])

    resultados["resultados"] = [
        {"puntaje": p, "probabilidad": q, "nivel_alerta": n}
        for p, q, n in zip(resultados["predictions"], resultados["probabilidades"], resultados["niveles_alerta"])
    ]
    logger.info(f"Niveles de alerta: {pd.Series(resultados['niveles_alerta']).value_counts().to_dict()}")
    return resultados


@api_router.get("/modelo", status_code=200)
def modelo() -> dict:
    """Ficha del modelo: periodos, métricas de validación y prueba, calibración, umbrales e importancia de variables.
    El tablero la usa en la vista de salud del modelo."""
    return model_metadata()
