
import glob
import pathlib
import re

import numpy as np
import pandas as pd

RAIZ = pathlib.Path(__file__).resolve().parent.parent
RAW = RAIZ / "data" / "raw"
OUT = RAIZ / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# Última vigencia completa de la etiqueta: el corte más reciente es de 2026.
ANIO_ETIQUETA_COMPLETO = 2025

BALANCE = {
    "activos_total": r"^Total de activos$",
    "activos_corrientes": r"^Activos corrientes totales$",
    "activos_no_corrientes": r"^Total de activos no corrientes$",
    "efectivo": r"^Efectivo y equivalentes al efectivo$",
    "inventarios": r"^Inventarios corrientes$",
    "cxc_corrientes": r"^Cuentas comerciales por cobrar y otras cuentas por cobrar corrientes$",
    "ppye": r"^Propiedades, planta y equipo$",
    "prop_inversion": r"^Propiedad.* de inversi",
    "pasivo_total": r"^Total pasivos$",
    "pasivo_corriente": r"^Pasivos corrientes totales$",
    "pasivo_no_corriente": r"^Total de pasivos no corrientes$",
    "patrimonio": r"^Patrimonio total$",
    "ganancias_acumuladas": r"^Ganancias acumuladas$",
    "capital": r"^Capital emitido$",
    "fin_corriente": r"^Otros pasivos financieros corrientes$",
    "fin_no_corriente": r"^Otros pasivos financieros no corrientes$",
    "prestamos_corriente": r"^Pr.stamos corrientes$",
    "prestamos_no_corriente": r"^Parte no corriente de pr.stamos no corrientes$",
    "prestamos_parte_corriente": r"^Parte corriente de pr.stamos no corrientes$",
    "cxp_corrientes": r"^Cuentas por pagar comerciales y otras cuentas por pagar corrientes$",
}
RESULTADOS = {
    "ingresos": r"^Ingresos de actividades ordinarias$",
    "costo_ventas": r"^Costo de ventas$",
    "utilidad_bruta": r"^Ganancia bruta$",
    "gastos_admin": r"^Gastos de administraci.n$",
    "utilidad_operacional": r"^Ganancia \(p.rdida\) por actividades de operaci.n$",
    "costos_financieros": r"^Costos financieros$",
    "utilidad_antes_impuestos": r"^Ganancia \(p.rdida\), antes de impuestos$",
    "utilidad_neta": r"^Ganancia \(p.rdida\)$",
}
ATRIBUTOS = {
    "estado_actual": r"^Estado actual$",
    "tipo_societario": r"^Tipo societario$",
    "fecha_constitucion": r"^Fecha de constituci",
    "ciudad": r"^Ciudad de la direcci.n del domicilio$",
    "departamento": r"^Departamento de la direcci.n del domicilio$",
    "razon_social": r"^Raz.n social de la sociedad$",
}


def mapear(df, reglas, col="concepto"):
    """Asigna un nombre corto a cada concepto (las tildes de la fuente vienen dañadas)."""
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    for nombre, patron in reglas.items():
        out[df[col].str.match(patron, na=False)] = nombre
    return out


def pivotar_estados(df, reglas):
    df = df.copy()
    df["var"] = mapear(df, reglas)
    df = df[df["var"].notna()]
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df["fecha_corte"] = pd.to_datetime(df["fecha_corte"])
    # periodo vigente: "Periodo Actual" (2018 en adelante) o el que empieza con el año del corte (2016-2017)
    vigente = (df["periodo"] == "Periodo Actual") | (df["periodo"].str[:4] == df["fecha_corte"].dt.year.astype(str))
    df = df[vigente]
    llaves = ["nit", "numero_radicado", "fecha_corte", "punto_entrada"]
    # propiedades de inversión trae dos variantes (costo / valor razonable): se suman
    return df.pivot_table(index=llaves, columns="var", values="valor", aggfunc="sum").reset_index()


def razon(a, b):
    r = a / b.where(b != 0)
    return r.replace([np.inf, -np.inf], np.nan)


def cargar_insolvencia():
    archivos = sorted(glob.glob(str(RAW / "insolvencia" / "insolvencia_*.xlsx")))
    if not archivos:
        return None, []
    partes, cortes = [], []
    for f in archivos:
        corte = re.search(r"insolvencia_(\d{4}-\d{2})", f).group(1)
        d = pd.read_excel(f)
        d.columns = [str(c).strip() for c in d.columns]
        col_anio = [c for c in d.columns if c.upper().startswith("A") and "APERTURA" in c.upper()][0]
        col_proc = [c for c in d.columns if c.upper() == "PROCESO"][0]
        d = d.rename(columns={col_anio: "anio_apertura", col_proc: "proceso"})
        d["nit"] = d["NIT"].astype(str).str.replace(r"\D", "", regex=True).str.lstrip("0")
        # el año viene como entero, como fecha (corte 2020-03) o con valores inválidos (ej. 5)
        num = pd.to_numeric(d["anio_apertura"], errors="coerce")
        fecha = pd.to_datetime(d["anio_apertura"].where(num.isna()), errors="coerce").dt.year
        anio = num.fillna(fecha)
        d["anio_apertura"] = anio.where(anio.between(1990, 2026))
        for c in ("CIIU", "NOMBRE", "proceso"):
            d[c] = d[c].astype(str).str.strip()
        d["corte"] = corte
        partes.append(d[["nit", "NOMBRE", "CIIU", "proceso", "anio_apertura", "corte"]])
        cortes.append(corte)
    ins = pd.concat(partes, ignore_index=True)
    ins = ins[ins["anio_apertura"].notna() & (ins["nit"] != "")]
    eventos = (ins.sort_values(["anio_apertura", "corte"])
                  .groupby("nit", as_index=False)
                  .agg(anio_apertura=("anio_apertura", "min"), proceso=("proceso", "first"),
                       ciiu=("CIIU", "first"), nombre=("NOMBRE", "first"),
                       n_cortes=("corte", "nunique"), primer_corte=("corte", "min")))
    eventos["anio_apertura"] = eventos["anio_apertura"].astype(int)
    return eventos, cortes


def main():
    ciiu = pd.read_parquet(RAW / "caratula_ciiu.parquet")
    attr = pd.read_parquet(RAW / "caratula_atributos.parquet")
    bal = pivotar_estados(pd.read_parquet(RAW / "balance_largo.parquet"), BALANCE)
    res = pivotar_estados(pd.read_parquet(RAW / "resultados_largo.parquet"), RESULTADOS)

    llaves = ["nit", "numero_radicado", "fecha_corte", "punto_entrada"]
    resumen = bal.merge(res, on=llaves, how="left")

    # un reporte por empresa-año
    resumen = resumen[resumen["fecha_corte"].dt.month == 12].copy()
    resumen["anio"] = resumen["fecha_corte"].dt.year
    resumen["prioridad"] = resumen["punto_entrada"].str.contains("Individual").map({True: 0, False: 1})
    resumen = (resumen.sort_values(["nit", "anio", "prioridad", "numero_radicado"],
                               ascending=[True, True, True, False])
                  .drop_duplicates(["nit", "anio"], keep="first")
                  .drop(columns="prioridad"))

    # CIIU del reporte
    ciiu["fecha_corte"] = pd.to_datetime(ciiu["fecha_corte"])
    c = (ciiu.sort_values("numero_radicado").drop_duplicates(["nit", "numero_radicado"], keep="last")
             [["nit", "numero_radicado", "ciiu"]])
    resumen = resumen.merge(c, on=["nit", "numero_radicado"], how="left")
    # si el radicado elegido no trae CIIU, se usa el último CIIU conocido del NIT
    ultimo = ciiu.sort_values("numero_radicado").groupby("nit")["ciiu"].last()
    resumen["ciiu"] = resumen["ciiu"].fillna(resumen["nit"].map(ultimo))
    resumen["ciiu_codigo"] = resumen["ciiu"].str.extract(r"^([A-Z]\d{4})")
    resumen["ciiu_grupo"] = resumen["ciiu_codigo"].str[:3].map(
        {"F41": "41 Construcción de edificios", "F42": "42 Obras de ingeniería civil",
         "F43": "43 Actividades especializadas", "L68": "68 Actividades inmobiliarias"})

    # atributos de la empresa (se toma el último reporte disponible de cada NIT)
    attr["var"] = mapear(attr, ATRIBUTOS)
    attr = attr[attr["var"].notna()].copy()
    attr["fecha_corte"] = pd.to_datetime(attr["fecha_corte"])
    attr = attr.sort_values(["nit", "numero_radicado"]).drop_duplicates(["nit", "var"], keep="last")
    attr = attr.pivot(index="nit", columns="var", values="valor").reset_index()
    resumen = resumen.merge(attr, on="nit", how="left")

    # razones financieras
    p = resumen
    p["deuda_financiera"] = p[["fin_corriente", "fin_no_corriente", "prestamos_corriente",
                               "prestamos_no_corriente", "prestamos_parte_corriente"]].sum(axis=1, min_count=1)
    p["liquidez_corriente"] = razon(p.activos_corrientes, p.pasivo_corriente)
    p["endeudamiento"] = razon(p.pasivo_total, p.activos_total)
    p["apalancamiento"] = razon(p.pasivo_total, p.patrimonio)
    p["deuda_fin_activos"] = razon(p.deuda_financiera, p.activos_total)
    p["pasivo_corriente_pasivo"] = razon(p.pasivo_corriente, p.pasivo_total)
    p["capital_trabajo_activos"] = razon(p.activos_corrientes - p.pasivo_corriente, p.activos_total)
    p["roa"] = razon(p.utilidad_neta, p.activos_total)
    p["roe"] = razon(p.utilidad_neta, p.patrimonio)
    p["margen_neto"] = razon(p.utilidad_neta, p.ingresos)
    p["margen_operacional"] = razon(p.utilidad_operacional, p.ingresos)
    p["cobertura_intereses"] = razon(p.utilidad_operacional, p.costos_financieros)
    p["rotacion_activos"] = razon(p.ingresos, p.activos_total)
    p["efectivo_activos"] = razon(p.efectivo, p.activos_total)
    p["ganancias_acum_activos"] = razon(p.ganancias_acumuladas, p.activos_total)
    p["ln_activos"] = np.log(p.activos_total.where(p.activos_total > 0))
    p["patrimonio_negativo"] = (p.patrimonio < 0).astype("Int8")
    p["perdida_neta"] = (p.utilidad_neta < 0).astype("Int8")
    # Z'' de Altman (empresas no manufactureras, mercados emergentes)
    p["altman_z2"] = (3.25 + 6.56 * p.capital_trabajo_activos + 3.26 * p.ganancias_acum_activos
                      + 6.72 * razon(p.utilidad_operacional, p.activos_total)
                      + 1.05 * razon(p.patrimonio, p.pasivo_total))

    # variable objetivo
    eventos, cortes = cargar_insolvencia()
    if eventos is None:
        print("AVISO: no hay cortes de insolvencia en data/raw/insolvencia/; se omite la variable objetivo.")
    else:
        print(f"Cortes de insolvencia usados ({len(cortes)}): {', '.join(cortes)}")
        eventos.to_parquet(OUT / "insolvencia_eventos.parquet", index=False)
        p = p.merge(eventos[["nit", "anio_apertura", "proceso"]], on="nit", how="left")
        p["post_evento"] = (p.anio_apertura.notna() & (p.anio >= p.anio_apertura)).astype("Int8")
        p["y_h1"] = ((p.anio_apertura == p.anio + 1)).astype("Int8")
        p["y_h2"] = ((p.anio_apertura == p.anio + 2)).astype("Int8")
        p["h1_valido"] = ((p.anio + 1) <= ANIO_ETIQUETA_COMPLETO).astype("Int8")
        p["h2_valido"] = ((p.anio + 2) <= ANIO_ETIQUETA_COMPLETO).astype("Int8")
    resumen = p.drop(columns=["ciiu"]).sort_values(["nit", "anio"]).reset_index(drop=True)
    resumen.to_parquet(OUT / "resumen_empresas.parquet", index=False)
    print(f"Resumen: {len(resumen):,} empresa-año, {resumen.nit.nunique():,} empresas, "
          f"años {resumen.anio.min()}-{resumen.anio.max()}")
    if "y_h1" in resumen:
        v = resumen[(resumen.post_evento == 0) & (resumen.h1_valido == 1)]
        print(f"Horizonte 1 (modelable): {len(v):,} filas, {int(v.y_h1.sum())} eventos "
              f"({v.y_h1.mean():.2%})")


if __name__ == "__main__":
    main()
