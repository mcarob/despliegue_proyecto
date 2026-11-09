"""Tablero de alerta temprana de insolvencia de constructoras e inmobiliarias (Entrega 2).

Cinco vistas, según la maqueta (mockup/): panorama del sector, lista de vigilancia, ficha de empresa,
simulador y salud del modelo.

Uso (desde la raíz del repositorio):
    pip install -r dashboard/requirements.txt
    python dashboard/app.py
y abrir http://127.0.0.1:8050 en el navegador.
"""

import json
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html, no_update

import modelo as M
from datos import SUBSECTORES, cargar_datos

# ---------------------------------------------------------------------------------------------------------
# Datos y modelo (se cargan una vez al iniciar)
# ---------------------------------------------------------------------------------------------------------
M.preparar()
DATOS = cargar_datos()
PUNTAJES = pd.read_csv(M.RUTA_PUNTAJES)
METRICAS = M.cargar_metricas()

BASE = DATOS[DATOS["post_evento"] == 0].merge(PUNTAJES, on=["nit", "anio"], how="inner")
BASE["puntaje100"] = BASE["puntaje"] * 100
ANIOS = sorted(BASE["anio"].unique())
ANIO_DEFECTO = 2024 if 2024 in ANIOS else ANIOS[-1]
DEPARTAMENTOS = sorted(BASE["departamento"].unique())

# Medianas por subsector y año (para la ficha y para explicar el puntaje)
MEDIANAS = BASE.groupby(["ciiu_grupo", "anio"])[M.RAZONES].median()

AZUL, ROJO, AMBAR, VERDE, GRIS = "#3f6ea5", "#b5332e", "#d9a23a", "#4f9a6a", "#9aa7b8"
COLOR_BANDA = {"Alta": ROJO, "Media": AMBAR, "Baja": VERDE}


def estilo_grafica(fig, alto=280):
    fig.update_traces(cliponaxis=False, selector=dict(type="bar"))
    fig.update_layout(height=alto, margin=dict(l=10, r=45, t=10, b=10), plot_bgcolor="#fff", paper_bgcolor="#fff",
                      font=dict(size=12), showlegend=False)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#e8ecf2")
    return fig


def filtrar(df, subsectores, departamento):
    if subsectores:
        df = df[df["ciiu_grupo"].isin(subsectores)]
    if departamento and departamento != "Todos":
        df = df[df["departamento"] == departamento]
    return df


@lru_cache(maxsize=32)
def lista_vigilancia(anio, subsectores, departamento, capacidad):
    """Empresas monitoreadas en el año: su último reporte de los últimos 3 años, ordenadas por puntaje.
    Banda alta = el % de capacidad de revisión; media = el 20 % siguiente; baja = el resto."""
    ventana = BASE[(BASE["anio"] <= anio) & (BASE["anio"] >= anio - 2)]
    ventana = filtrar(ventana, list(subsectores), departamento)
    ultimo = ventana.sort_values("anio").groupby("nit").tail(1).copy()
    ultimo = ultimo.sort_values("puntaje", ascending=False).reset_index(drop=True)
    n = len(ultimo)
    ultimo["puesto"] = np.arange(1, n + 1)
    corte_alta = int(np.ceil(n * capacidad / 100))
    corte_media = int(np.ceil(n * (capacidad + 20) / 100))
    ultimo["banda"] = np.where(ultimo["puesto"] <= corte_alta, "Alta",
                               np.where(ultimo["puesto"] <= corte_media, "Media", "Baja"))
    ultimo["sin_reporte"] = ultimo["anio"] < anio
    # Cambio frente al puntaje de la misma empresa el año anterior a su último reporte
    previo = BASE[["nit", "anio", "puntaje100"]].rename(columns={"puntaje100": "previo"})
    previo["anio"] = previo["anio"] + 1
    ultimo = ultimo.merge(previo, on=["nit", "anio"], how="left")
    ultimo["cambio"] = ultimo["puntaje100"] - ultimo["previo"]
    umbral = float(ultimo.loc[ultimo["puesto"] == corte_alta, "puntaje"].iloc[0]) if corte_alta else 1.0
    return ultimo, umbral


def factor_principal(filas):
    """Variable que más empuja el puntaje de cada empresa (aporte frente a la mediana de su subsector)."""
    registros, ids = [], []
    for i, fila in filas.iterrows():
        base = fila[M.VARIABLES].to_dict()
        registros.append(base)
        med = MEDIANAS.loc[(fila["ciiu_grupo"], fila["anio"])]
        for v in M.RAZONES:
            alterna = dict(base)
            alterna[v] = med[v]
            registros.append(alterna)
        ids.append(i)
    if not registros:
        return pd.Series(dtype=str)
    p = M.calcular_riesgo(registros).reshape(len(ids), len(M.RAZONES) + 1)
    aportes = p[:, [0]] - p[:, 1:]
    mejores = aportes.argmax(axis=1)
    textos = []
    for k, i in enumerate(ids):
        v = M.RAZONES[mejores[k]]
        textos.append(f"{M.ETIQUETAS[v]} {filas.loc[i, v]:.2f}" if aportes[k, mejores[k]] > 0 else "—")
    return pd.Series(textos, index=ids)


def kpi(titulo, valor, nota, clase=""):
    return html.Div(className="kpi", children=[
        html.Div(titulo, className="t"), html.Div(valor, className=f"v {clase}"), html.Div(nota, className="n")])


def fmt(x, dec=2):
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ---------------------------------------------------------------------------------------------------------
# Estructura general: barra lateral, filtros globales y contenido de cada vista
# ---------------------------------------------------------------------------------------------------------
VISTAS = [("/", "1. Panorama del sector"), ("/lista", "2. Lista de vigilancia"), ("/ficha", "3. Ficha de empresa"),
          ("/simulador", "4. Simulador"), ("/salud", "5. Salud del modelo")]

app = Dash(__name__, suppress_callback_exceptions=True, title="Alerta temprana de insolvencia")

app.layout = html.Div(className="contenedor", children=[
    dcc.Location(id="url"),
    dcc.Store(id="empresa", data=None),
    html.Div(className="barra", children=[
        html.H2("Alerta temprana"),
        html.Div("Insolvencia de constructoras e inmobiliarias", className="sub"),
        html.Div(id="menu"),
        html.Div(className="pie", children=[
            f"Modelo {METRICAS['version']} · {METRICAS['modelo']}", html.Br(),
            "Datos: NIIF Supersociedades", html.Br(), f"Años {ANIOS[0]}-{ANIOS[-1]}"]),
    ]),
    html.Div(className="principal", children=[
        html.Div(id="encabezado"),
        html.Div(className="filtros", children=[
            html.Div(className="filtro", children=["Año fiscal", dcc.Dropdown(
                id="f-anio", options=[{"label": str(a), "value": int(a)} for a in ANIOS], value=int(ANIO_DEFECTO),
                clearable=False)]),
            html.Div(className="filtro ancho", children=["Subsector", dcc.Dropdown(
                id="f-sub", options=[{"label": v, "value": k} for k, v in SUBSECTORES.items()],
                value=list(SUBSECTORES), multi=True)]),
            html.Div(className="filtro", children=["Departamento", dcc.Dropdown(
                id="f-depto", options=["Todos"] + DEPARTAMENTOS, value="Todos", clearable=False)]),
            html.Div(className="filtro", children=["Capacidad de revisión", dcc.Dropdown(
                id="f-cap", options=[{"label": f"{c} %", "value": c} for c in (5, 10, 15, 20)], value=10,
                clearable=False)]),
        ]),
        dcc.Loading(html.Div(id="contenido")),
    ]),
])

ENCABEZADOS = {
    "/": ("Panorama del sector",
          "Cuántas empresas se monitorean, cuántas entran en insolvencia y dónde se concentra el riesgo"),
    "/lista": ("Lista de vigilancia",
               "A qué empresas revisar primero: ordenadas por riesgo y limitadas a la capacidad de revisión"),
    "/ficha": ("Ficha de empresa", "Por qué una empresa está en la lista y qué pedirle en la revisión"),
    "/simulador": ("Simulador",
                   "Evaluar una empresa nueva o un escenario: el tablero envía las razones financieras al modelo"),
    "/salud": ("Salud del modelo",
               "Saber si la lista sigue siendo confiable: desempeño frente a las líneas base, deriva y versión"),
}


@app.callback(Output("menu", "children"), Output("encabezado", "children"), Input("url", "pathname"))
def menu(ruta):
    ruta = ruta if ruta in ENCABEZADOS else "/"
    enlaces = [dcc.Link(t, href=r, className="activo" if r == ruta else "") for r, t in VISTAS]
    titulo, desc = ENCABEZADOS[ruta]
    return enlaces, [html.H1(titulo), html.Div(desc, className="desc")]


@app.callback(Output("contenido", "children"),
              Input("url", "pathname"), Input("f-anio", "value"), Input("f-sub", "value"),
              Input("f-depto", "value"), Input("f-cap", "value"), State("empresa", "data"))
def mostrar(ruta, anio, subs, depto, cap, empresa):
    subs = tuple(sorted(subs or SUBSECTORES))
    if ruta == "/lista":
        return vista_lista(anio, subs, depto, cap)
    if ruta == "/ficha":
        return vista_ficha(anio, subs, depto, cap, empresa)
    if ruta == "/simulador":
        return vista_simulador(anio, subs, depto, cap, empresa)
    if ruta == "/salud":
        return vista_salud()
    return vista_panorama(anio, subs, depto, cap)


# ---------------------------------------------------------------------------------------------------------
# 1. Panorama del sector
# ---------------------------------------------------------------------------------------------------------
def vista_panorama(anio, subs, depto, cap):
    d = filtrar(BASE, list(subs), depto)
    etiquetados = d[d["h1_valido"] == 1]
    lista, _ = lista_vigilancia(anio, subs, depto, cap)

    del_anio = d[d["anio"] == anio]
    eventos = int(etiquetados["y_h1"].sum())
    anios_etiq = f"{int(etiquetados['anio'].min())}-{int(etiquetados['anio'].max())}" if len(etiquetados) else "—"
    tasa_anio = etiquetados.loc[etiquetados["anio"] == anio, "y_h1"]
    tasa_txt = f"{fmt(tasa_anio.mean() * 1000, 1)} por mil" if len(tasa_anio) else "sin etiqueta aún"
    alta = int((lista["banda"] == "Alta").sum())

    por_anio = etiquetados.groupby("anio")["y_h1"].mean() * 1000
    f1 = go.Figure(go.Bar(x=por_anio.index.astype(str), y=por_anio.values, marker_color=[
        "#1f3556" if a == anio else AZUL for a in por_anio.index], text=[fmt(v, 1) for v in por_anio.values],
        textposition="outside"))
    estilo_grafica(f1).update_yaxes(title="por mil empresas")

    por_sub = etiquetados.groupby("subsector")["y_h1"].mean().sort_values() * 1000
    f2 = go.Figure(go.Bar(x=por_sub.values, y=por_sub.index, orientation="h", marker_color=[ROJO, AMBAR, VERDE][
        :len(por_sub)][::-1], text=[fmt(v, 1) for v in por_sub.values], textposition="outside"))
    estilo_grafica(f2, 200).update_xaxes(title="por mil empresas")

    conteo = lista["banda"].value_counts().reindex(["Alta", "Media", "Baja"]).fillna(0)
    f3 = go.Figure([go.Bar(x=[conteo[b]], y=["Empresas"], orientation="h", name=b, marker_color=COLOR_BANDA[b],
                           text=f"{b}: {int(conteo[b]):,}".replace(",", "."), textposition="inside")
                    for b in conteo.index])
    estilo_grafica(f3, 120).update_layout(barmode="stack")
    f3.update_yaxes(showticklabels=False)

    deptos = lista[lista["banda"] == "Alta"]["departamento"].value_counts().head(8).sort_values()
    f4 = go.Figure(go.Bar(x=deptos.values, y=deptos.index, orientation="h", marker_color="#1f3556",
                          text=deptos.values, textposition="outside"))
    estilo_grafica(f4, 260)

    return [
        html.Div(className="kpis", children=[
            kpi("Empresas monitoreadas", f"{len(lista):,}".replace(",", "."),
                f"con estados financieros {anio - 2}-{anio}"),
            kpi("Procesos de insolvencia", f"{eventos:,}".replace(",", "."),
                f"eventos a un año, años fiscales {anios_etiq}"),
            kpi(f"Tasa de insolvencia {anio}", tasa_txt, "empresa-año, horizonte de 1 año"),
            kpi("En banda de riesgo alta", f"{alta:,}".replace(",", "."), f"top {cap} % según el modelo", "rojo"),
        ]),
        html.Div(className="fila", children=[
            html.Div(className="tarjeta", children=[
                html.H3("Tasa de insolvencia por año (por mil empresas)"),
                html.Div("Proporción de empresas que abren un proceso de insolvencia al año siguiente",
                         className="desc"), dcc.Graph(figure=f1, config={"displayModeBar": False})]),
            html.Div(className="tarjeta", children=[
                html.H3("Tasa por subsector (por mil empresas)"),
                html.Div(f"Promedio {anios_etiq}. Las obras civiles (42) quedan por fuera del universo",
                         className="desc"), dcc.Graph(figure=f2, config={"displayModeBar": False})]),
        ]),
        html.Div(className="fila", children=[
            html.Div(className="tarjeta", children=[
                html.H3(f"Empresas por banda de riesgo (año fiscal {anio})"),
                html.Div(f"Alta = {cap} % de mayor riesgo (capacidad de revisión); media = el 20 % siguiente",
                         className="desc"), dcc.Graph(figure=f3, config={"displayModeBar": False})]),
            html.Div(className="tarjeta", children=[
                html.H3("Empresas en riesgo alto por departamento"),
                dcc.Graph(figure=f4, config={"displayModeBar": False})]),
        ]),
        html.Div("Tasas calculadas con resumen_empresas (CIIU 41, 43 y 68; sin años posteriores al evento). "
                 "Las bandas salen del modelo random forest de la Entrega 2.", className="nota"),
    ]


# ---------------------------------------------------------------------------------------------------------
# 2. Lista de vigilancia
# ---------------------------------------------------------------------------------------------------------
def vista_lista(anio, subs, depto, cap):
    lista, umbral = lista_vigilancia(anio, subs, depto, cap)
    revisar = lista[lista["banda"] == "Alta"].copy()
    revisar["factor"] = factor_principal(revisar)
    tabla = pd.DataFrame({
        "#": revisar["puesto"], "Empresa": revisar["codigo"], "Subsector": revisar["subsector"],
        "Departamento": revisar["departamento"], "Puntaje": revisar["puntaje100"].round(1),
        "Banda": revisar["banda"],
        "Cambio": revisar["cambio"].map(lambda c: "Nueva" if pd.isna(c) else f"{c:+.1f} pts"),
        "Factor principal": np.where(revisar["sin_reporte"], "Sin reporte reciente", revisar["factor"]),
        "Último reporte": revisar["anio"].astype(int),
    })
    eventos_valid = lista[lista["h1_valido"] == 1]
    captura_txt = ""
    if eventos_valid["y_h1"].sum() > 0:
        cap_real = eventos_valid.loc[eventos_valid["banda"] == "Alta", "y_h1"].sum() / eventos_valid["y_h1"].sum()
        captura_txt = (f" En {anio}, esta lista habría contenido el {fmt(cap_real * 100, 0)} % de las empresas que "
                       "abrieron insolvencia al año siguiente.")
    return [
        html.Div(className="tarjeta", children=[
            html.H3(f"Revisar {len(revisar):,} de {len(lista):,} empresas ({cap} %)".replace(",", ".")),
            html.Div(f"Umbral de vigilancia: puntaje ≥ {fmt(umbral * 100, 1)}.{captura_txt}", className="desc"),
            html.Div(className="aviso", children=(
                "«Sin reporte reciente»: la empresa no reportó estados financieros en el año elegido y se califica "
                "con su último reporte. No reportar es una señal: falta el balance en el 14,8 % de las insolventes "
                "frente al 7,9 % de las sanas.")),
            dash_table.DataTable(
                id="tabla-lista", data=tabla.to_dict("records"), columns=[{"name": c, "id": c} for c in tabla.columns],
                page_size=15, sort_action="native", filter_action="native",
                style_cell={"fontSize": 13, "padding": "6px", "textAlign": "left", "fontFamily": "Segoe UI, Arial"},
                style_header={"fontWeight": "600", "backgroundColor": "#f3f5f8"},
                style_data_conditional=[
                    {"if": {"filter_query": "{Banda} = Alta", "column_id": "Banda"}, "color": ROJO,
                     "fontWeight": "600"},
                    {"if": {"filter_query": "{Factor principal} = 'Sin reporte reciente'",
                            "column_id": "Factor principal"}, "color": "#c68a12", "fontWeight": "600"},
                    {"if": {"column_id": "Empresa"}, "color": AZUL, "fontWeight": "600", "cursor": "pointer"},
                ]),
            html.Div("Haga clic en el código de una empresa para abrir su ficha. Los identificadores son anónimos "
                     "(hash del NIT): el tablero no muestra razones sociales. El puntaje (0-100) ordena el riesgo; "
                     "no es una probabilidad calibrada.", className="nota"),
        ]),
    ]


@app.callback(Output("empresa", "data"), Output("url", "pathname"), Input("tabla-lista", "active_cell"),
              State("tabla-lista", "derived_viewport_data"), prevent_initial_call=True)
def abrir_ficha(celda, filas):
    if not celda or celda["column_id"] != "Empresa" or not filas:
        return no_update, no_update
    return filas[celda["row"]]["Empresa"], "/ficha"


# ---------------------------------------------------------------------------------------------------------
# 3. Ficha de empresa
# ---------------------------------------------------------------------------------------------------------
def vista_ficha(anio, subs, depto, cap, empresa):
    lista, _ = lista_vigilancia(anio, subs, depto, cap)
    opciones = [{"label": f"{r.codigo} · puesto {r.puesto} · {r.banda}", "value": r.codigo}
                for r in lista.head(500).itertuples()]
    if empresa not in set(lista["codigo"]):
        empresa = lista["codigo"].iloc[0]
        opciones_val = empresa
    else:
        opciones_val = empresa
        if empresa not in {o["value"] for o in opciones}:
            r = lista[lista["codigo"] == empresa].iloc[0]
            opciones.insert(0, {"label": f"{r.codigo} · puesto {r.puesto} · {r.banda}", "value": r.codigo})
    return [
        html.Div(className="filtros", children=[html.Div(className="filtro ancho", children=[
            "Empresa (ordenadas por riesgo)", dcc.Dropdown(id="ficha-empresa", options=opciones, value=opciones_val,
                                                           clearable=False)])]),
        html.Div(id="ficha-detalle", children=detalle_ficha(opciones_val, anio, subs, depto, cap)),
    ]


@app.callback(Output("ficha-detalle", "children"), Output("empresa", "data", allow_duplicate=True),
              Input("ficha-empresa", "value"), State("f-anio", "value"), State("f-sub", "value"),
              State("f-depto", "value"), State("f-cap", "value"), prevent_initial_call=True)
def cambiar_empresa(codigo, anio, subs, depto, cap):
    subs = tuple(sorted(subs or SUBSECTORES))
    return detalle_ficha(codigo, anio, subs, depto, cap), codigo


def detalle_ficha(codigo, anio, subs, depto, cap):
    lista, _ = lista_vigilancia(anio, subs, depto, cap)
    r = lista[lista["codigo"] == codigo].iloc[0]
    historia = BASE[BASE["codigo"] == codigo].sort_values("anio")
    med = MEDIANAS.loc[r["ciiu_grupo"]]

    aportes = M.explicar(r, MEDIANAS.loc[(r["ciiu_grupo"], r["anio"])].to_dict()).head(7)[::-1] * 100
    f_aporte = go.Figure(go.Bar(x=aportes.values, y=[M.ETIQUETAS[v] for v in aportes.index], orientation="h",
                                marker_color=[ROJO if v > 0 else VERDE for v in aportes.values],
                                text=[f"{v:+.1f}" for v in aportes.values], textposition="outside"))
    estilo_grafica(f_aporte, 260).update_xaxes(title="puntos de puntaje")

    graficas = []
    for v in ["endeudamiento", "efectivo_activos", "roa", "altman_z2"]:
        f = go.Figure([
            go.Scatter(x=historia["anio"], y=historia[v], mode="lines+markers", line=dict(color=ROJO), name="Empresa"),
            go.Scatter(x=med.index, y=med[v], mode="lines", line=dict(color=GRIS, dash="dash"),
                       name="Mediana del subsector"),
        ])
        estilo_grafica(f, 170)
        actual = historia[v].iloc[-1] if len(historia) else np.nan
        graficas.append(html.Div(children=[
            html.Div([html.B(M.ETIQUETAS[v]), html.Span(f"  Actual: {fmt(actual)}" if pd.notna(actual) else "",
                                                          className="rojo")], style={"fontSize": 13}),
            dcc.Graph(figure=f, config={"displayModeBar": False})]))

    z = r["altman_z2"]
    zona = "zona de peligro" if z < 1.1 else ("zona gris" if z < 2.6 else "zona segura")
    clase_z = "rojo" if z < 1.1 else ("ambar" if z < 2.6 else "verde")
    return [
        html.Div(className="fila", children=[
            html.Div(className="tarjeta", children=[
                html.Div(f"Lista de vigilancia › {codigo}", className="nota"),
                html.H1(codigo),
                html.Div(f"{r['subsector']} · {r['departamento']} · último reporte {int(r['anio'])}",
                         className="desc"),
                html.Div("Puntaje de riesgo", className="nota"),
                html.Div([html.Span(fmt(r["puntaje100"], 1), style={"fontSize": 40, "fontWeight": 700},
                                    className="rojo" if r["banda"] == "Alta" else ""),
                          html.Span(" / 100  "), html.Span(r["banda"], className=f"chip {r['banda']}")]),
                html.Div(f"Puesto {int(r['puesto']):,} de {len(lista):,}".replace(",", ".") +
                         ("" if pd.isna(r["cambio"]) else f" · cambio frente al año anterior: {r['cambio']:+.1f} pts"),
                         className="desc"),
                html.Div("Sin reporte reciente: se califica con su último reporte disponible." if r["sin_reporte"]
                         else "", className="aviso" if r["sin_reporte"] else ""),
            ]),
            html.Div(className="tarjeta", children=[
                html.H3("Qué explica el puntaje"),
                html.Div("Cuánto cambiaría el puntaje si cada variable tomara la mediana de su subsector "
                         "(rojo aumenta el riesgo, verde lo reduce)", className="desc"),
                dcc.Graph(figure=f_aporte, config={"displayModeBar": False})]),
        ]),
        html.Div(className="fila", children=[
            html.Div(className="tarjeta", children=[
                html.H3("Evolución de razones clave frente a la mediana del subsector"),
                html.Div("Línea roja: la empresa · línea gris: mediana del subsector", className="desc"),
                html.Div(graficas, style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "6px"})]),
            html.Div(className="tarjeta", children=[
                html.H3("Z'' de Altman y guía de revisión"),
                html.Div([html.Span(f"Z'' = {fmt(z)}: ", className=clase_z), html.Span(zona, className=clase_z)],
                         style={"fontSize": 15, "fontWeight": 600, "margin": "6px 0"}),
                html.Div("< 1,1 peligro · 1,1 a 2,6 gris · > 2,6 segura", className="nota"),
                html.H3("Qué pedir en la revisión", style={"marginTop": 14}),
                html.Ul(className="guia", children=[
                    html.Li("Flujo de caja proyectado a 12 meses"),
                    html.Li("Detalle y vencimientos de las obligaciones financieras"),
                    html.Li("Estado de preventas y obras en curso"),
                    html.Li("Estados financieros parciales del año en curso"),
                    html.Li("Garantías y procesos judiciales vigentes"),
                ]),
            ]),
        ]),
    ]


# ---------------------------------------------------------------------------------------------------------
# 4. Simulador
# ---------------------------------------------------------------------------------------------------------
CAMPOS_SIM = ["endeudamiento", "liquidez_corriente", "efectivo_activos", "deuda_fin_activos", "roa",
              "margen_operacional", "margen_neto", "rotacion_activos", "capital_trabajo_activos",
              "ganancias_acum_activos", "pasivo_corriente_pasivo", "apalancamiento", "roe", "cobertura_intereses",
              "ln_activos", "altman_z2"]


def valores_empresa(codigo):
    fila = BASE[BASE["codigo"] == codigo].sort_values("anio")
    return fila.iloc[-1] if len(fila) else None


def vista_simulador(anio, subs, depto, cap, empresa):
    lista, _ = lista_vigilancia(anio, subs, depto, cap)
    codigo = empresa if empresa in set(BASE["codigo"]) else lista["codigo"].iloc[0]
    r = valores_empresa(codigo)
    campos = [html.Label([M.ETIQUETAS[v], dcc.Input(id=f"sim-{v}", type="number", debounce=True,
                                                    value=None if pd.isna(r[v]) else round(float(r[v]), 3))])
              for v in CAMPOS_SIM]
    return [html.Div(className="fila mitad", children=[
        html.Div(className="tarjeta", children=[
            html.H3("Datos de la empresa"),
            html.Div(f"Precargados con el último reporte de {codigo}. Puede cambiarlos para evaluar una empresa "
                     "nueva.", className="desc"),
            html.Label(["Subsector (CIIU)", dcc.Dropdown(id="sim-ciiu", options=[
                {"label": v, "value": k} for k, v in SUBSECTORES.items()], value=r["ciiu_grupo"],
                clearable=False)], style={"fontSize": 12, "color": "#5b6b80"}),
            html.Div(className="formulario", style={"marginTop": 8}, children=campos),
            dcc.Checklist(id="sim-ind", options=[{"label": " Patrimonio negativo", "value": "patrimonio_negativo"},
                                                 {"label": " Pérdida neta en el año", "value": "perdida_neta"}],
                          value=[v for v in M.INDICADORES if r[v] == 1], style={"fontSize": 13, "margin": "10px 0"}),
            html.H3("Escenario a comparar", style={"marginTop": 6}),
            html.Div("Por ejemplo: la empresa se capitaliza y reduce su deuda", className="desc"),
            html.Div(className="formulario", children=[
                html.Label(["Endeudamiento en el escenario", dcc.Input(
                    id="esc-endeudamiento", type="number", value=round(float(r["endeudamiento"]) * 0.8, 3)
                    if pd.notna(r["endeudamiento"]) else None)]),
                html.Label(["Deuda financiera / activos en el escenario", dcc.Input(
                    id="esc-deuda", type="number", value=round(float(r["deuda_fin_activos"]) * 0.7, 3)
                    if pd.notna(r["deuda_fin_activos"]) else None)]),
            ]),
            html.Div(style={"marginTop": 12}, children=[html.Button("Calcular riesgo", id="sim-calcular",
                                                                    className="boton")]),
            html.Div("Campos vacíos = dato faltante (el modelo usa la mediana de entrenamiento).", className="nota"),
        ]),
        html.Div(className="tarjeta", children=[html.Div(id="sim-resultado", children=html.Div(
            "Pulse «Calcular riesgo» para ver el resultado del modelo.", className="desc"))]),
    ])]


@app.callback(Output("sim-resultado", "children"), Input("sim-calcular", "n_clicks"),
              [State(f"sim-{v}", "value") for v in CAMPOS_SIM] +
              [State("sim-ciiu", "value"), State("sim-ind", "value"), State("esc-endeudamiento", "value"),
               State("esc-deuda", "value"), State("f-anio", "value"), State("f-sub", "value"),
               State("f-depto", "value"), State("f-cap", "value")],
              prevent_initial_call=True)
def simular(_, *valores):
    razones = dict(zip(CAMPOS_SIM, valores[:len(CAMPOS_SIM)]))
    ciiu, indicadores, esc_end, esc_deuda, anio, subs, depto, cap = valores[len(CAMPOS_SIM):]
    subs = tuple(sorted(subs or SUBSECTORES))
    registro = {**{v: (np.nan if razones[v] is None else razones[v]) for v in CAMPOS_SIM},
                **{v: int(v in (indicadores or [])) for v in M.INDICADORES}, "ciiu_grupo": ciiu}
    escenario = dict(registro)
    if esc_end is not None:
        escenario["endeudamiento"] = esc_end
    if esc_deuda is not None:
        escenario["deuda_fin_activos"] = esc_deuda
    p_actual, p_esc = M.calcular_riesgo([registro, escenario])

    lista, umbral = lista_vigilancia(anio, subs, depto, cap)
    percentil = (lista["puntaje"] < p_actual).mean() * 100
    puesto = int((lista["puntaje"] > p_actual).sum()) + 1
    n = len(lista)
    banda = "Alta" if puesto <= np.ceil(n * cap / 100) else ("Media" if puesto <= np.ceil(n * (cap + 20) / 100)
                                                             else "Baja")
    en_vigilancia = p_actual >= umbral
    entrada = {"inputs": [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in registro.items()}]}
    salida = {"puntaje": round(float(p_actual), 4), "banda": banda, "en_vigilancia": bool(en_vigilancia),
              "umbral": round(umbral, 4), "version": METRICAS["version"]}

    escala = go.Figure(go.Bar(x=[100], y=[""], orientation="h", marker_color="#e8ecf2"))
    escala.add_vline(x=umbral * 100, line_color="#1f3556", line_dash="dash",
                     annotation_text=f"umbral {fmt(umbral * 100, 1)}")
    escala.add_trace(go.Scatter(x=[p_actual * 100], y=[""], mode="markers", marker=dict(size=16, color=ROJO)))
    estilo_grafica(escala, 90).update_xaxes(range=[0, 100])

    return [
        html.H3("Resultado del modelo"),
        html.Div(className="kpis", style={"gridTemplateColumns": "1fr 1fr 1fr"}, children=[
            kpi("Puntaje de riesgo", fmt(p_actual * 100, 1), "escala 0-100", "rojo" if banda == "Alta" else ""),
            kpi("Banda", banda, f"capacidad de revisión {cap} %",
                {"Alta": "rojo", "Media": "ambar", "Baja": "verde"}[banda]),
            kpi("Puesto estimado", f"{puesto:,} de {n:,}".replace(",", "."), f"percentil {fmt(percentil, 1)}"),
        ]),
        dcc.Graph(figure=escala, config={"displayModeBar": False}),
        html.Div(className="alerta" if en_vigilancia else "ok", children=(
            "Riesgo alto: la empresa entraría en la lista de vigilancia." if en_vigilancia else
            "La empresa no entraría en la lista de vigilancia con esta capacidad de revisión.")),
        html.H3("Comparar con el escenario", style={"marginTop": 14}),
        dash_table.DataTable(
            data=[{"Variable": "Endeudamiento", "Actual": registro["endeudamiento"],
                   "Escenario": escenario["endeudamiento"]},
                  {"Variable": "Deuda financiera / activos", "Actual": registro["deuda_fin_activos"],
                   "Escenario": escenario["deuda_fin_activos"]},
                  {"Variable": "Puntaje de riesgo", "Actual": round(p_actual * 100, 1),
                   "Escenario": round(p_esc * 100, 1)}],
            style_cell={"fontSize": 13, "padding": "5px", "fontFamily": "Segoe UI, Arial"},
            style_header={"fontWeight": "600", "backgroundColor": "#f3f5f8"}),
        html.Div(f"Efecto del escenario: {(p_esc - p_actual) * 100:+.1f} puntos.", className="desc",
                 style={"marginTop": 6}),
        html.H3("Formato de la llamada al modelo", style={"marginTop": 14}),
        html.Div("En esta entrega el tablero usa el modelo directamente; en la entrega final enviará esta misma "
                 "entrada a la API (POST /predict).", className="desc"),
        html.Pre(className="json", children="POST /predict\n" + json.dumps(entrada, ensure_ascii=False) +
                 "\n\nRespuesta\n" + json.dumps(salida, ensure_ascii=False)),
    ]


# ---------------------------------------------------------------------------------------------------------
# 5. Salud del modelo
# ---------------------------------------------------------------------------------------------------------
def vista_salud():
    comp = METRICAS["comparacion"]
    nombre_modelo = [k for k in comp if k.startswith("Modelo")][0]
    m = comp[nombre_modelo]
    alt = comp["Z'' de Altman (línea base)"]
    end = comp["Endeudamiento (línea base)"]
    prev = METRICAS["prevalencia_prueba"]

    tabla = [{"Modelo": k, "AUC": fmt(v["auc"]), "AUC-PR": fmt(v["auc_pr"], 3),
              "Eventos capturados al revisar el 10 %": f"{fmt(v['captura_top10'] * 100, 1)} %"}
             for k, v in comp.items()]

    curva = go.Figure()
    for nombre, color in (("Modelo", "#1f3556"), ("Z'' de Altman", AMBAR)):
        x, y = METRICAS["curvas"][nombre]
        curva.add_trace(go.Scatter(x=np.array(x) * 100, y=np.array(y) * 100, mode="lines", name=nombre,
                                   line=dict(color=color, width=3)))
    curva.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Al azar",
                               line=dict(color=GRIS, dash="dash")))
    curva.add_vline(x=10, line_color=ROJO, line_dash="dot", annotation_text="Capacidad 10 %")
    estilo_grafica(curva, 300).update_layout(showlegend=True, legend=dict(x=0.6, y=0.1))
    curva.update_xaxes(title="% de empresas revisadas")
    curva.update_yaxes(title="% de eventos capturados")

    psi = pd.Series(METRICAS["psi"]).sort_values()
    f_psi = go.Figure(go.Bar(x=psi.values, y=[M.ETIQUETAS[v] for v in psi.index], orientation="h",
                             marker_color=[VERDE if v < 0.1 else (AMBAR if v < 0.25 else ROJO) for v in psi.values],
                             text=[fmt(v) for v in psi.values], textposition="outside"))
    f_psi.add_vline(x=0.1, line_color=AMBAR, line_dash="dot")
    f_psi.add_vline(x=0.25, line_color=ROJO, line_dash="dot")
    estilo_grafica(f_psi, 260)

    imp = pd.Series(METRICAS["importancia"]).sort_values(ascending=False).head(5)
    top = ", ".join(M.ETIQUETAS.get(v, v.replace("ciiu_", "subsector ")) for v in imp.index)

    return [
        html.Div(className="kpis", children=[
            kpi("AUC", fmt(m["auc"]), f"Z'' de Altman {fmt(alt['auc'])} · endeudamiento {fmt(end['auc'])}", "verde"),
            kpi("AUC-PR", fmt(m["auc_pr"], 3), f"{fmt(m['auc_pr'] / prev, 1)} veces la prevalencia "
                                               f"({fmt(prev * 100)} %)", "verde"),
            kpi("Brier", fmt(METRICAS["brier"], 4), "probabilidades sin calibrar (class_weight)", "ambar"),
            kpi("Eventos capturados al revisar el 10 %", f"{fmt(m['captura_top10'] * 100, 1)} %",
                f"Z'' de Altman: {fmt(alt['captura_top10'] * 100, 1)} %", "verde"),
        ]),
        html.Div(className="fila", children=[
            html.Div(className="tarjeta", children=[
                html.H3("Comparación con las líneas base (prueba fuera de tiempo)"),
                html.Div(f"Entrenado con {METRICAS['entrenamiento']} y evaluado en {METRICAS['prueba']} "
                         f"({METRICAS['filas_prueba']:,} empresa-año, {METRICAS['eventos_prueba']} eventos)"
                         .replace(",", "."), className="desc"),
                dash_table.DataTable(data=tabla, columns=[{"name": c, "id": c} for c in tabla[0]],
                                     style_cell={"fontSize": 13, "padding": "6px", "fontFamily": "Segoe UI, Arial"},
                                     style_header={"fontWeight": "600", "backgroundColor": "#f3f5f8"},
                                     style_data_conditional=[{"if": {"row_index": len(tabla) - 1},
                                                              "fontWeight": "700", "color": "#2f7d4f"}]),
                html.Div("El modelo supera ambas líneas base en todas las métricas.", className="ok"),
            ]),
            html.Div(className="tarjeta", children=[
                html.H3("Curva de captura de eventos"),
                dcc.Graph(figure=curva, config={"displayModeBar": False})]),
        ]),
        html.Div(className="fila", children=[
            html.Div(className="tarjeta", children=[
                html.H3(f"Deriva de las variables (PSI {METRICAS['psi_anio']} frente a entrenamiento)"),
                html.Div("PSI menor que 0,10 estable · 0,10 a 0,25 vigilar · mayor que 0,25 reentrenar",
                         className="desc"),
                dcc.Graph(figure=f_psi, config={"displayModeBar": False})]),
            html.Div(className="tarjeta", children=[
                html.H3("Datos y versión del modelo"),
                html.Ul(className="guia", children=[
                    html.Li(f"Versión: {METRICAS['version']} ({METRICAS['modelo']}, registrado en MLflow, "
                            "experimento alerta-insolvencia)"),
                    html.Li(f"Entrenamiento: {METRICAS['entrenamiento']}, {METRICAS['filas_entrenamiento']:,} "
                            f"empresa-año y {METRICAS['eventos_entrenamiento']} eventos".replace(",", ".")),
                    html.Li(f"Prueba: {METRICAS['prueba']}"),
                    html.Li("Objetivo: apertura de insolvencia al año siguiente (y_h1), CIIU 41, 43 y 68"),
                    html.Li(f"Variables con más peso: {top}"),
                ]),
                html.Div(className="aviso", children=(
                    "Las probabilidades del modelo están infladas por el balanceo de clases: sirven para ordenar el "
                    "riesgo, no como probabilidad real. Por eso el tablero muestra puntaje, puesto y banda.")),
            ]),
        ]),
    ]


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
