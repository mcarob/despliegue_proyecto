# Alerta temprana de insolvencia de constructoras e inmobiliarias

Proyecto del curso MIID-4304 Despliegue de Soluciones Analíticas (MIAD, Uniandes, 2026-15).

**Pregunta de negocio.** ¿Cómo puede un banco identificar, con al menos un año de anticipación y con una
capacidad de revisión limitada, las constructoras e inmobiliarias que entrarán en un proceso de
insolvencia, para actuar a tiempo y reducir la pérdida y la provisión de su cartera del sector?

**Prototipo objetivo.** Modelos supervisados empaquetados, una API de inferencia y un tablero (lista de
vigilancia y ficha de empresa), desplegados con contenedores Docker. Este repositorio corresponde a la
Entrega 1 (semana 3): datos versionados con DVC, exploración de datos y maqueta.

## Estructura

```
data/raw/            datos crudos (versionados con DVC, no con Git)
data/processed/      panel empresa-año (versionado con DVC)
src/                 scripts de extracción y construcción del panel
notebooks/           exploración de datos
mockup/              maqueta HTML del tablero
docs/                diccionario de datos y figuras
```

## Fuentes de datos

| Fuente | Contenido | Acceso |
|---|---|---|
| Supersociedades, Estados Financieros NIIF: Estado de Situación Financiera (`pfdp-zks5`) | Balance en formato largo (concepto-valor) | API SODA de datos.gov.co, CC BY-SA 4.0 |
| Supersociedades, Estados Financieros NIIF: Estado de Resultado Integral (`prwj-nzxa`) | Resultados en formato largo | API SODA de datos.gov.co, CC BY-SA 4.0 |
| Supersociedades, Estados Financieros NIIF: Carátula (`6hqw-m3dm`) | CIIU, ciudad, estado, tipo societario | API SODA de datos.gov.co, CC BY-SA 4.0 |
| Supersociedades, Base de datos de insolvencia (Excel por corte) | Procesos de insolvencia en curso y año de apertura | [Página de informes](https://www.supersociedades.gov.co/es/web/procedimientos-de-insolvencia/informes), descarga manual |

Las URLs de los cortes de insolvencia están en `src/insolvencia_urls.json`. Los archivos se guardan como
`data/raw/insolvencia/insolvencia_AAAA-MM.xlsx`.

## Reproducir

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
dvc pull                          # trae los datos versionados (requiere acceso al remoto)
# o bien, regenerarlos desde las fuentes:
python src/extract_niif.py        # estados financieros del sector (CIIU 41, 42, 43 y 68)
python src/build_panel.py         # panel empresa-año con razones y variable objetivo
```

## Variable objetivo

`y_h1 = 1` si el proceso de insolvencia se abre en el año siguiente al de los estados financieros;
`y_h2 = 1` si se abre dos años después. Los años posteriores a la apertura se excluyen (`post_evento`).
Detalle en `docs/diccionario_datos.md`.

## Equipo

Ver el reporte de trabajo en equipo de la entrega. Cada integrante trabaja en su rama y fusiona con
pull request.
