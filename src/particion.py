
from __future__ import annotations

import pathlib

import pandas as pd

RAIZ = pathlib.Path(__file__).resolve().parents[1]
RUTA_RESUMEN = RAIZ / "data" / "processed" / "resumen_empresas.parquet"

NUCLEO = ["41 Construcción de edificios", "43 Actividades especializadas", "68 Actividades inmobiliarias"]

ANIO_ENTRENAMIENTO_HASTA = 2021
ANIO_VALIDACION = 2022
ANIOS_PRUEBA = (2023, 2024)

RAZONES = [
    "liquidez_corriente", "endeudamiento", "apalancamiento", "deuda_fin_activos", "roa", "margen_operacional",
    "cobertura_intereses", "efectivo_activos", "capital_trabajo_activos", "ganancias_acum_activos", "ln_activos",
    "altman_z2",
]
BINARIAS = ["patrimonio_negativo", "perdida_neta"]
FEATURES = RAZONES + BINARIAS  
CATEGORICAS = ["ciiu_grupo"]  
PROHIBIDAS = {"anio_apertura", "post_evento", "y_h1", "y_h2", "h1_valido", "h2_valido", "y"}


def cargar_modelable(horizonte: int = 1, ruta: pathlib.Path = RUTA_RESUMEN) -> pd.DataFrame:
    if horizonte not in (1, 2):
        raise ValueError("horizonte debe ser 1 o 2")
    p = pd.read_parquet(ruta)
    p = p[p["ciiu_grupo"].isin(NUCLEO)]
    p = p[(p["post_evento"] == 0) & (p[f"h{horizonte}_valido"] == 1)].copy()
    p["y"] = p[f"y_h{horizonte}"].astype(int)
    assert not (PROHIBIDAS & set(FEATURES)), "una variable explicativa usa la etiqueta o el futuro"
    return p.reset_index(drop=True)


def particion(horizonte: int = 1, purga: bool = True, ruta: pathlib.Path = RUTA_RESUMEN):
    d = cargar_modelable(horizonte, ruta)
    hasta = ANIO_ENTRENAMIENTO_HASTA
    if purga:
        hasta = min(hasta, ANIO_VALIDACION - horizonte)
    train = d[d["anio"] <= hasta]
    valid = d[d["anio"] == ANIO_VALIDACION]
    test = d[d["anio"].isin(ANIOS_PRUEBA)]
    return train, valid, test


def resumen(horizonte: int = 1, purga: bool = True) -> pd.DataFrame:
    filas = []
    for nombre, df in zip(("entrenamiento", "validación", "prueba"), particion(horizonte, purga)):
        filas.append({"horizonte": horizonte, "conjunto": nombre, "años": f"{df.anio.min()}-{df.anio.max()}",
                      "empresa-año": len(df), "empresas": df.nit.nunique(), "eventos": int(df.y.sum()),
                      "prevalencia %": round(df.y.mean() * 100, 2)})
    return pd.DataFrame(filas)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    for h in (1, 2):
        print(resumen(h, purga=True).to_string(index=False))
        print("__"*30)
