"""Pruebas del modelo empaquetado: que prediga, que lo haga bien y que las salidas tengan sentido."""

import math

import numpy as np
from sklearn.metrics import roc_auc_score

from alerta_insolvencia import __version__
from alerta_insolvencia.predict import make_prediction, model_metadata


def test_make_prediction(sample_input_data):
    # Dado: la muestra de 2.000 empresas; Cuando: se predice; Entonces: un resultado por empresa y sin errores
    result = make_prediction(input_data=sample_input_data)

    predictions = result.get("predictions")
    assert isinstance(predictions, list)
    assert result.get("errors") is None
    assert result.get("version") == __version__
    assert len(predictions) == len(sample_input_data) == 2000
    assert len(result["probabilidades"]) == len(result["niveles_alerta"]) == 2000
    assert all(0.0 <= p <= 1.0 for p in predictions)


def test_modelo_ordena_bien_el_riesgo(sample_input_data):
    # Prueba de regresión de calidad: si un cambio daña el modelo, el AUC en la muestra cae y la prueba falla
    result = make_prediction(input_data=sample_input_data)
    auc = roc_auc_score(sample_input_data["y_h1"], result["predictions"])
    assert auc > 0.85


def test_probabilidad_calibrada_en_escala_real(sample_input_data):
    result = make_prediction(input_data=sample_input_data)
    puntaje = np.array(result["predictions"])
    probabilidad = np.array(result["probabilidades"])
    # La calibración es monótona: no cambia el orden de las empresas
    orden = np.argsort(puntaje)
    assert np.all(np.diff(probabilidad[orden]) >= -1e-12)
    # Y baja las probabilidades infladas por el balanceo de clases a la escala de la tasa real (menos de 1 %)
    assert probabilidad.mean() < puntaje.mean()
    assert probabilidad.mean() < 0.05


def test_niveles_de_alerta(sample_input_data):
    result = make_prediction(input_data=sample_input_data)
    niveles = set(result["niveles_alerta"])
    assert niveles <= {"alto", "medio", "bajo"}
    umbrales = model_metadata()["umbrales_alerta"]
    for puntaje, nivel in zip(result["predictions"], result["niveles_alerta"]):
        if puntaje >= umbrales["alto"]:
            assert nivel == "alto"
        elif puntaje < umbrales["medio"]:
            assert nivel == "bajo"


def test_un_solo_registro_con_faltantes(registro_valido):
    # Un dato faltante (None) se imputa con la mediana de entrenamiento en vez de rechazar la empresa
    registro_valido["cobertura_intereses"] = None
    result = make_prediction(input_data=[registro_valido])
    assert result["errors"] is None
    assert len(result["predictions"]) == 1
    assert not math.isnan(result["predictions"][0])


def test_metadatos_completos():
    meta = model_metadata()
    assert meta["version"] == __version__
    for clave in ("metricas_validacion", "metricas_prueba", "intervalos_95_prueba", "calibracion",
                  "umbrales_alerta", "importancia", "lineas_base_prueba"):
        assert clave in meta
    # El modelo debe superar la línea base de Altman en la prueba fuera de tiempo
    assert meta["metricas_prueba"]["auc"] > meta["lineas_base_prueba"]["altman_z2"]["auc"]
