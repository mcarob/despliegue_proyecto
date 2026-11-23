"""Pruebas de la API con el cliente de pruebas de FastAPI (no necesitan levantar el servidor)."""

import numpy as np
import pandas as pd
from alerta_insolvencia.config.core import config
from fastapi.testclient import TestClient

from app.schemas.predict import MultipleDataInputs

URL = "/api/v1"


def payload(df: pd.DataFrame) -> dict:
    """Convierte filas a la entrada de la API: NaN pasa a null (dato faltante)."""
    datos = df[config.modelo.variables].astype(object).replace({np.nan: None})
    return {"inputs": datos.to_dict(orient="records")}


def test_health(client: TestClient) -> None:
    respuesta = client.get(f"{URL}/health")
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["model_version"] == "2.0.0"
    assert set(cuerpo) == {"name", "api_version", "model_version"}


def test_make_prediction(client: TestClient, test_data: pd.DataFrame) -> None:
    respuesta = client.post(f"{URL}/predict", json=payload(test_data))
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["errors"] is None
    assert len(cuerpo["predictions"]) == len(cuerpo["resultados"]) == len(test_data)
    primero = cuerpo["resultados"][0]
    assert 0 <= primero["probabilidad"] <= primero["puntaje"] <= 1
    assert primero["nivel_alerta"] in {"alto", "medio", "bajo"}


def test_ejemplo_de_la_documentacion(client: TestClient) -> None:
    # El ejemplo que aparece en /docs debe funcionar y ordenar bien: la empresa en problemas va primero
    ejemplo = MultipleDataInputs.model_config["json_schema_extra"]["example"]
    respuesta = client.post(f"{URL}/predict", json=ejemplo)
    assert respuesta.status_code == 200
    riesgosa, sana = respuesta.json()["resultados"]
    assert riesgosa["puntaje"] > sana["puntaje"]
    assert riesgosa["nivel_alerta"] == "alto"


def test_subsector_invalido_responde_422(client: TestClient, test_data: pd.DataFrame) -> None:
    cuerpo = payload(test_data.head(1))
    cuerpo["inputs"][0]["ciiu_grupo"] = "42"
    respuesta = client.post(f"{URL}/predict", json=cuerpo)
    assert respuesta.status_code == 422


def test_nan_o_infinito_responde_422_y_no_500(client: TestClient) -> None:
    # requests y httpx escriben NaN e Infinity tal cual; la API debe decir qué campo está mal, no fallar
    for valor in ("NaN", "Infinity", "1e400"):
        cuerpo = '{"inputs": [{"roa": %s, "ciiu_grupo": "41"}]}' % valor
        respuesta = client.post(f"{URL}/predict", content=cuerpo, headers={"Content-Type": "application/json"})
        assert respuesta.status_code == 422, valor
        assert respuesta.json()["detail"][0]["loc"][-1] == "roa"


def test_lote_vacio_responde_400(client: TestClient) -> None:
    respuesta = client.post(f"{URL}/predict", json={"inputs": []})
    assert respuesta.status_code == 400


def test_ficha_del_modelo(client: TestClient) -> None:
    respuesta = client.get(f"{URL}/modelo")
    assert respuesta.status_code == 200
    ficha = respuesta.json()
    assert ficha["version"] == "2.0.0"
    assert ficha["metricas_prueba"]["auc"] > 0.85


def test_pagina_de_inicio(client: TestClient) -> None:
    respuesta = client.get("/")
    assert respuesta.status_code == 200
    assert "/docs" in respuesta.text
