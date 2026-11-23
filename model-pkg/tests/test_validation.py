"""Pruebas de la validación de entradas: qué se acepta, qué se rechaza y con qué mensaje."""

import pandas as pd

from alerta_insolvencia.predict import make_prediction


def test_rechaza_subsector_fuera_del_universo(registro_valido):
    registro_valido["ciiu_grupo"] = "42"  # obras civiles: fuera del alcance del modelo
    result = make_prediction(input_data=[registro_valido])
    assert result["predictions"] is None
    assert result["errors"][0]["loc"][-1] == "ciiu_grupo"


def test_acepta_codigo_ciiu_completo(registro_valido):
    registro_valido["ciiu_grupo"] = "4111"  # el código de 4 dígitos se reduce al grupo 41
    assert make_prediction(input_data=[registro_valido])["errors"] is None


def test_rechaza_texto_en_una_razon(registro_valido):
    registro_valido["endeudamiento"] = "alto"
    result = make_prediction(input_data=[registro_valido])
    assert result["predictions"] is None
    assert result["errors"][0]["loc"][-1] == "endeudamiento"


def test_rechaza_infinito(registro_valido):
    registro_valido["liquidez_corriente"] = float("inf")
    assert make_prediction(input_data=[registro_valido])["errors"] is not None


def test_rechaza_indicador_que_no_es_cero_o_uno(registro_valido):
    registro_valido["perdida_neta"] = 3
    assert make_prediction(input_data=[registro_valido])["errors"] is not None


def test_columnas_faltantes_y_sobrantes(registro_valido):
    # Solo el subsector es obligatorio: lo que falte se imputa y lo que sobre se ignora
    result = make_prediction(input_data=[{"ciiu_grupo": "68", "endeudamiento": 1.2, "columna_extra": "x"}])
    assert result["errors"] is None
    assert len(result["predictions"]) == 1


def test_acepta_un_solo_registro_como_diccionario(registro_valido):
    result = make_prediction(input_data=registro_valido)
    assert result["errors"] is None
    assert len(result["predictions"]) == 1


def test_subsector_ausente_es_campo_obligatorio():
    result = make_prediction(input_data=[{"endeudamiento": 0.9}])
    assert result["predictions"] is None
    assert result["errors"][0]["type"] == "missing"
    assert result["errors"][0]["loc"][-1] == "ciiu_grupo"


def test_rechaza_columnas_repetidas(registro_valido):
    datos = pd.DataFrame([registro_valido])
    datos = pd.concat([datos, datos[["roa"]]], axis=1)  # dos columnas "roa"
    result = make_prediction(input_data=datos)
    assert result["predictions"] is None
    assert result["errors"][0]["type"] == "columnas_repetidas"


def test_lote_vacio():
    result = make_prediction(input_data=[])
    assert result["predictions"] is None
    assert result["errors"][0]["type"] == "tamano_lote"
