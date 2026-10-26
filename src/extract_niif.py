"""
EXTRAE ALGUNOS ARCHIVOS NIIF
"""
import argparse
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

BASE = "https://www.datos.gov.co/resource"
RAIZ = pathlib.Path(__file__).resolve().parent.parent
RAW = RAIZ / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
PAGINA = 50000
DESDE = "2016-01-01T00:00:00.000"

# En la fuente las tildes vienen dañadas (U+FFFD), por eso se usa LIKE con %.
CONCEPTOS_BALANCE = [
    "Total de activos", "Activos corrientes totales", "Total de activos no corrientes",
    "Efectivo y equivalentes al efectivo",
    "Inventarios corrientes",
    "Cuentas comerciales por cobrar y otras cuentas por cobrar corrientes",
    "Propiedades, planta y equipo",
    "Propiedad% de inversi%",
    "Total pasivos", "Pasivos corrientes totales", "Total de pasivos no corrientes",
    "Patrimonio total", "Ganancias acumuladas", "Capital emitido",
    "Otros pasivos financieros corrientes", "Otros pasivos financieros no corrientes",
    "Pr%stamos corrientes", "Parte no corriente de pr%stamos no corrientes",
    "Parte corriente de pr%stamos no corrientes",
    "Cuentas por pagar comerciales y otras cuentas por pagar corrientes",
]
CONCEPTOS_RESULTADOS = [
    "Ingresos de actividades ordinarias", "Costo de ventas", "Ganancia bruta",
    "Gastos de administraci%n", "Ganancia (p%rdida) por actividades de operaci%n",
    "Costos financieros", "Ganancia (p%rdida), antes de impuestos", "Ganancia (p%rdida)",
]
CONCEPTOS_CARATULA = [
    "Estado actual", "Tipo societario", "Fecha de constituci%",
    "Ciudad de la direcci%n del domicilio", "Departamento de la direcci%n del domicilio",
    "Raz%n social de la sociedad",
]
CIIU_SECTOR = ["F41", "F42", "F43", "L68"]


def get(ds, params, intentos=5):
    for i in range(intentos):
        try:
            r = requests.get(f"{BASE}/{ds}.json", params=params, timeout=180)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code}")
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if i == intentos - 1:
                raise
            time.sleep(2 ** (i + 1))


def paginar(ds, where, select):
    """Una sola consulta sin orden (rápida); solo si llena la página se pagina con orden estable."""
    lote = get(ds, {"$where": where, "$select": select, "$limit": PAGINA})
    if len(lote) < PAGINA:
        return pd.DataFrame(lote)
    filas, offset = [], 0
    while True:
        lote = get(ds, {"$where": where, "$select": select, "$order": ":id",
                        "$limit": PAGINA, "$offset": offset})
        filas += lote
        if len(lote) < PAGINA:
            return pd.DataFrame(filas)
        offset += PAGINA


def or_like(campo, patrones):
    return "(" + " OR ".join(f"{campo} like '{p}'" for p in patrones) + ")"


def por_lotes(ds, nits, concepto_like, extra, select, tam, nombre):
    def tarea(lote):
        lista = ",".join(f"'{n}'" for n in lote)
        where = (f"nit in ({lista}) AND fecha_corte >= '{DESDE}' AND {extra} "
                 f"AND {or_like('concepto', concepto_like)}")
        return paginar(ds, where, select)
    lotes = [nits[i:i + tam] for i in range(0, len(nits), tam)]
    with ThreadPoolExecutor(max_workers=4) as ex:
        partes = []
        for k, df in enumerate(ex.map(tarea, lotes), 1):
            partes.append(df)
            if k % 10 == 0 or k == len(lotes):
                print(f"  {nombre}: lote {k}/{len(lotes)}", flush=True)
    return pd.concat(partes, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-nits", type=int, default=None)
    ap.add_argument("--rehacer", action="store_true", help="vuelve a descargar aunque el archivo exista")
    a = ap.parse_args()
    hecho = lambda f: (RAW / f).exists() and not a.rehacer and not a.max_nits

    print("1/4 Caratula: filings del sector por CIIU")
    if hecho("caratula_ciiu.parquet"):
        ciiu = pd.read_parquet(RAW / "caratula_ciiu.parquet")
    else:
        where = (f"fecha_corte >= '{DESDE}' AND concepto like 'Clasificaci%Industrial%' AND "
                 + or_like("valor", [f"{c}%" for c in CIIU_SECTOR]))
        ciiu = paginar("6hqw-m3dm", where,
                       "nit,numero_radicado,fecha_corte,punto_entrada,valor")
        ciiu = ciiu.rename(columns={"valor": "ciiu"})
    print(f"  {len(ciiu):,} filings, {ciiu.nit.nunique():,} NIT")
    nits = sorted(ciiu.nit.unique())
    if a.max_nits:
        nits = nits[:a.max_nits]
        ciiu = ciiu[ciiu.nit.isin(nits)]
    ciiu.to_parquet(RAW / "caratula_ciiu.parquet", index=False)

    print("2/4 Caratula: atributos de las empresas")
    if not hecho("caratula_atributos.parquet"):
        attr = por_lotes("6hqw-m3dm", nits, CONCEPTOS_CARATULA, "true",
                         "nit,numero_radicado,fecha_corte,concepto,valor", 150, "atributos")
        attr.to_parquet(RAW / "caratula_atributos.parquet", index=False)

    # Desde 2018 el periodo vigente se rotula "Periodo Actual"; en 2016-2017 se rotula con el año
    # ("2016-dic-31" en el balance, "2016" en resultados). El filtro fino se hace en build_resumen_empresas.py.
    filtro = ("(periodo='Periodo Actual' OR periodo in ('2016','2017','2016-dic-31','2017-dic-31')) "
              "AND (punto_entrada like '%Individual%' OR punto_entrada like '%Separado%')")
    sel = "nit,numero_radicado,fecha_corte,punto_entrada,periodo,concepto,valor"
    print("3/4 Estado de situacion financiera")
    if not hecho("balance_largo.parquet"):
        bal = por_lotes("pfdp-zks5", nits, CONCEPTOS_BALANCE, filtro, sel, 60, "balance")
        bal.to_parquet(RAW / "balance_largo.parquet", index=False)
    print("4/4 Estado de resultado integral")
    if not hecho("resultados_largo.parquet"):
        res = por_lotes("prwj-nzxa", nits, CONCEPTOS_RESULTADOS, filtro, sel, 60, "resultados")
        res.to_parquet(RAW / "resultados_largo.parquet", index=False)
    b, r = pd.read_parquet(RAW / "balance_largo.parquet"), pd.read_parquet(RAW / "resultados_largo.parquet")
    print(f"Listo: balance {len(b):,} filas, resultados {len(r):,} filas")


if __name__ == "__main__":
    main()
