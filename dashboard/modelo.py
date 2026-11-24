"""Conexión del tablero con la API de inferencia (api/ del repositorio).

El tablero no tiene el modelo: le envía las razones financieras a la API (POST /api/v1/predict) y recibe el
puntaje de riesgo, la probabilidad calibrada y el nivel de alerta de cada empresa. La ficha del modelo (métricas,
umbrales, curva de captura e importancia de variables) la trae de GET /api/v1/modelo.

La dirección de la API se indica con la variable de entorno API_URL (por defecto http://localhost:8001).
"""

import json
import os
import time

import numpy as np
import pandas as pd
import requests

from datos import CARPETA_TABLERO, cargar_datos

API_URL = os.environ.get("API_URL", "http://localhost:8001").rstrip("/")
LOTE = 5000  # la API acepta hasta 5.000 empresas por llamada

CARPETA_ARTEFACTOS = CARPETA_TABLERO / "artefactos"
RUTA_FICHA = CARPETA_ARTEFACTOS / "ficha_modelo.json"

RAZONES = [
    "liquidez_corriente", "endeudamiento", "apalancamiento", "deuda_fin_activos",
    "pasivo_corriente_pasivo", "capital_trabajo_activos", "roa", "roe", "margen_neto",
    "margen_operacional", "cobertura_intereses", "rotacion_activos", "efectivo_activos",
    "ganancias_acum_activos", "ln_activos", "altman_z2",
]
INDICADORES = ["patrimonio_negativo", "perdida_neta"]
VARIABLES = RAZONES + INDICADORES + ["ciiu_grupo"]
ANIOS_ENTRENAMIENTO = range(2016, 2022)

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
    "ciiu_41": "Subsector 41", "ciiu_43": "Subsector 43", "ciiu_68": "Subsector 68",
}


# ---------------------------------------------------------------------------------------------------------
# Llamadas a la API
# ---------------------------------------------------------------------------------------------------------
def esperar_api(intentos=60, pausa=5):
    """Espera a que la API responda en /health (útil cuando los dos contenedores arrancan al tiempo)."""
    for i in range(intentos):
        try:
            r = requests.get(f"{API_URL}/api/v1/health", timeout=5)
            if r.ok:
                return r.json()
        except requests.RequestException:
            pass
        print(f"Esperando la API en {API_URL} ({i + 1}/{intentos})...")
        time.sleep(pausa)
    raise RuntimeError(f"La API no responde en {API_URL}. Revise que esté encendida y la variable API_URL.")


def _limpiar(registros):
    """Prepara los registros para JSON: los faltantes (NaN) van como null y los indicadores como enteros."""
    limpios = []
    for r in registros:
        fila = {}
        for v in VARIABLES:
            x = r.get(v)
            if v == "ciiu_grupo":
                fila[v] = str(x)[:2]
            elif x is None or (isinstance(x, float) and not np.isfinite(x)):
                fila[v] = None
            elif v in INDICADORES:
                fila[v] = int(x)
            else:
                fila[v] = float(x)
        limpios.append(fila)
    return limpios


def predecir(registros):
    """Envía las empresas a la API en lotes y devuelve un DataFrame con puntaje, probabilidad y nivel de alerta,
    en el mismo orden de la entrada."""
    registros = _limpiar(registros)
    partes = []
    for i in range(0, len(registros), LOTE):
        r = requests.post(f"{API_URL}/api/v1/predict", json={"inputs": registros[i:i + LOTE]}, timeout=120)
        r.raise_for_status()
        partes.extend(r.json()["resultados"])
    return pd.DataFrame(partes, columns=["puntaje", "probabilidad", "nivel_alerta"])


def calcular_riesgo(registros):
    """Puntaje de riesgo (0 a 1) de cada empresa, calculado por la API."""
    return predecir(registros)["puntaje"].to_numpy()


def llamada_api(registros):
    """Hace una llamada a /predict y devuelve la entrada y la respuesta tal como viajan (para mostrarlas en el
    simulador)."""
    entrada = {"inputs": _limpiar(registros)}
    r = requests.post(f"{API_URL}/api/v1/predict", json=entrada, timeout=60)
    r.raise_for_status()
    return entrada, r.json()


def ficha_modelo():
    r = requests.get(f"{API_URL}/api/v1/modelo", timeout=30)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------------------------------------
# Preparación al iniciar: puntajes de todas las empresas-año y métricas para la vista de salud
# ---------------------------------------------------------------------------------------------------------
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


def curva_captura(y, puntaje, pasos=50):
    orden = np.argsort(-puntaje)
    acumulado = np.cumsum(y[orden]) / y.sum()
    x = np.linspace(0, 1, pasos + 1)
    idx = np.clip((x * len(y)).astype(int) - 1, 0, len(y) - 1)
    return [0.0] + list(x[1:]), [0.0] + list(acumulado[idx[1:]])


def preparar():
    """Pide a la API el puntaje de todas las empresas-año y guarda la ficha del modelo. Se repite solo si cambia
    la versión del modelo que sirve la API."""
    estado = esperar_api()
    version = estado["model_version"]
    ruta_puntajes = CARPETA_ARTEFACTOS / f"puntajes_modelo_{version}.csv"
    if ruta_puntajes.exists() and RUTA_FICHA.exists():
        return
    CARPETA_ARTEFACTOS.mkdir(exist_ok=True)
    print(f"Calculando puntajes con la API (modelo {version}); solo la primera vez...")

    d = cargar_datos()
    modelable = d[d["post_evento"] == 0]
    resultado = predecir(modelable[VARIABLES].to_dict("records"))
    puntajes = modelable[["nit", "anio"]].reset_index(drop=True).join(resultado)
    puntajes.to_csv(ruta_puntajes, index=False)

    ficha = ficha_modelo()
    # Curva de captura del Z'' de Altman en la misma prueba, para compararla con la del modelo
    prueba = modelable[modelable["anio"].between(2022, 2024) & (modelable["h1_valido"] == 1)]
    altman = -prueba["altman_z2"].fillna(prueba["altman_z2"].median()).to_numpy()
    ficha["curva_altman_prueba"] = curva_captura(prueba["y_h1"].to_numpy(), altman)
    entrenamiento = modelable[modelable["anio"].isin(ANIOS_ENTRENAMIENTO)]
    ultimo = int(modelable["anio"].max())
    ficha["psi_anio"] = ultimo
    ficha["psi"] = {v: psi(entrenamiento[v], modelable.loc[modelable["anio"] == ultimo, v])
                    for v in ["endeudamiento", "liquidez_corriente", "efectivo_activos", "deuda_fin_activos",
                              "roa", "margen_operacional"]}
    RUTA_FICHA.write_text(json.dumps(ficha, ensure_ascii=False, indent=2), encoding="utf-8")
    m = ficha["metricas_prueba"]
    print(f"Listo. Modelo {version}: AUC {m['auc']:.3f}, captura en el 10 % {m['captura_top10']:.1%}")


def cargar_puntajes():
    version = esperar_api()["model_version"]
    # el NIT se lee como texto: coincide con el de resumen_empresas, sea .parquet o .csv
    return pd.read_csv(CARPETA_ARTEFACTOS / f"puntajes_modelo_{version}.csv", dtype={"nit": str})


def cargar_metricas():
    return json.loads(RUTA_FICHA.read_text(encoding="utf-8"))


def explicar(fila, medianas):
    """Aporte aproximado de cada variable: cuánto baja o sube el puntaje si esa variable tomara el valor mediano
    de su subsector (las demás quedan iguales). Positivo = la variable aumenta el riesgo."""
    base = fila[VARIABLES].to_dict()
    variantes = [base]
    for v in RAZONES:
        alterna = dict(base)
        alterna[v] = medianas.get(v, np.nan)
        variantes.append(alterna)
    p = calcular_riesgo(variantes)
    return pd.Series(p[0] - p[1:], index=RAZONES).sort_values(key=abs, ascending=False)
