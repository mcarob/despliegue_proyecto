"""Lectura de la tabla resumen_empresas para el tablero.

Los datos no están en GitHub: se obtienen con `dvc pull` (remoto S3 del proyecto) o copiando el archivo
resumen_empresas (.parquet o .csv) en la carpeta data/processed/ del repositorio.
"""

import hashlib
import pathlib

import pandas as pd

CARPETA_TABLERO = pathlib.Path(__file__).resolve().parent
RAIZ = CARPETA_TABLERO.parent
CARPETA_DATOS = RAIZ / "data" / "processed"

# Universo del proyecto: las obras civiles (42) quedan por fuera, como pidió el tutor en la Entrega 1
SUBSECTORES = {
    "41": "41 Construcción de edificios",
    "43": "43 Actividades especializadas",
    "68": "68 Actividades inmobiliarias",
}


def ruta_datos():
    for nombre in ("resumen_empresas.parquet", "resumen_empresas.csv"):
        ruta = CARPETA_DATOS / nombre
        if ruta.exists():
            return ruta
    raise FileNotFoundError(
        f"No se encontró resumen_empresas en {CARPETA_DATOS}. Ejecute `dvc pull` o copie allí "
        "resumen_empresas.parquet o resumen_empresas.csv."
    )


def codigo_anonimo(nit):
    """Identificador anónimo de la empresa: el tablero no muestra razones sociales."""
    return "E-" + hashlib.sha256(str(nit).encode()).hexdigest()[:6].upper()


def cargar_datos():
    ruta = ruta_datos()
    d = pd.read_parquet(ruta) if ruta.suffix == ".parquet" else pd.read_csv(ruta, low_memory=False)
    d["ciiu_grupo"] = d["ciiu_grupo"].astype(str).str[:2]
    d = d[d["ciiu_grupo"].isin(SUBSECTORES)].copy()
    d["subsector"] = d["ciiu_grupo"].map(SUBSECTORES)
    d["departamento"] = d["departamento"].fillna("SIN DATO").str.title()
    d["codigo"] = d["nit"].map(codigo_anonimo)
    return d.reset_index(drop=True)
