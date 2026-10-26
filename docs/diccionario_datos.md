# Diccionario de datos

Unidad de análisis: **empresa-año** (NIT y año fiscal, con corte a 31 de diciembre).
Universo: sociedades con estados financieros NIIF ante la Superintendencia de Sociedades y CIIU Rev. 4 A.C.
F41 (construcción de edificios), F43 (actividades especializadas de construcción) y L68 (actividades
inmobiliarias). Las obras civiles (F42) se conservan para pruebas de sensibilidad.

Los valores monetarios están en la unidad reportada por la fuente (la carátula indica la unidad de
diligenciamiento; la Guía de planteamiento los trata como miles de pesos). No se deflactan en el panel.

## Archivos

| Archivo | Descripción | Origen |
|---|---|---|
| `data/raw/caratula_ciiu.parquet` | Reportes del sector con su CIIU | `6hqw-m3dm` |
| `data/raw/caratula_atributos.parquet` | Ciudad, departamento, tipo societario, estado y razón social (formato largo) | `6hqw-m3dm` |
| `data/raw/balance_largo.parquet` | Conceptos del estado de situación financiera (formato largo) | `pfdp-zks5` |
| `data/raw/resultados_largo.parquet` | Conceptos del estado de resultado integral (formato largo) | `prwj-nzxa` |
| `data/raw/insolvencia/*.xlsx` | Cortes de la base de procesos de insolvencia en curso | Supersociedades |
| `data/processed/panel_empresa_anio.parquet` | Panel con una fila por empresa-año | `src/build_panel.py` |
| `data/processed/insolvencia_eventos.parquet` | Un registro por NIT con el año de apertura más temprano | `src/build_panel.py` |

## Reglas de construcción del panel

1. Se conservan solo los estados financieros con corte a 31 de diciembre y los puntos de entrada
   *Individual* o *Separado*, periodo actual.
2. Un reporte por empresa-año: Individual antes que Separado y, a igualdad, el último radicado.
3. Las tildes de los nombres de concepto vienen dañadas en la fuente; se emparejan con expresiones regulares.
4. El NIT se normaliza a dígitos sin dígito de verificación para cruzar con la base de insolvencia.

## Variable objetivo

| Variable | Definición |
|---|---|
| `anio_apertura` | Año de apertura del proceso (columna `AÑO_DE_APERTURA/CELEBRACIÓN`), el menor entre los cortes |
| `y_h1` | 1 si `anio_apertura = anio + 1` (horizonte de un año) |
| `y_h2` | 1 si `anio_apertura = anio + 2` (horizonte de dos años) |
| `post_evento` | 1 si `anio >= anio_apertura`; se excluye para modelar (evita fugas de información) |
| `h1_valido`, `h2_valido` | 1 si el año de apertura buscado está dentro del periodo con etiqueta completa (hasta 2025) |

La base de insolvencia publica solo los procesos **en curso** en cada corte. Los procesos abiertos y
terminados entre dos cortes no aparecen; por eso se descargan varios cortes y se toma la unión.

## Variables del panel

| Variable | Descripción |
|---|---|
| `nit`, `anio`, `numero_radicado`, `punto_entrada` | Identificación del reporte |
| `ciiu_codigo`, `ciiu_grupo` | Actividad económica (ej. F4111) y grupo (41, 42, 43, 68) |
| `razon_social`, `ciudad`, `departamento`, `tipo_societario`, `estado_actual`, `fecha_constitucion` | Atributos de la carátula (último reporte del NIT) |
| `activos_total`, `activos_corrientes`, `activos_no_corrientes`, `efectivo`, `inventarios`, `cxc_corrientes`, `ppye`, `prop_inversion` | Balance, activos |
| `pasivo_total`, `pasivo_corriente`, `pasivo_no_corriente`, `cxp_corrientes`, `patrimonio`, `capital`, `ganancias_acumuladas` | Balance, pasivos y patrimonio |
| `fin_corriente`, `fin_no_corriente`, `prestamos_corriente`, `prestamos_no_corriente`, `prestamos_parte_corriente` | Pasivos financieros |
| `deuda_financiera` | Suma de los pasivos financieros (aproxima la exposición) |
| `ingresos`, `costo_ventas`, `utilidad_bruta`, `gastos_admin`, `utilidad_operacional`, `costos_financieros`, `utilidad_antes_impuestos`, `utilidad_neta` | Estado de resultados |
| `liquidez_corriente`, `endeudamiento`, `apalancamiento`, `deuda_fin_activos`, `pasivo_corriente_pasivo`, `capital_trabajo_activos` | Razones de liquidez y endeudamiento |
| `roa`, `roe`, `margen_neto`, `margen_operacional`, `cobertura_intereses`, `rotacion_activos`, `efectivo_activos`, `ganancias_acum_activos`, `ln_activos` | Razones de rentabilidad, cobertura y tamaño |
| `patrimonio_negativo`, `perdida_neta` | Indicadores binarios |
| `altman_z2` | Z'' de Altman = 3,25 + 6,56·X1 + 3,26·X2 + 6,72·X3 + 1,05·X4, con X1 = capital de trabajo / activos, X2 = ganancias acumuladas / activos, X3 = utilidad operacional / activos, X4 = patrimonio / pasivos |

Una razón con denominador cero queda como valor faltante.

## Tamaño y periodo de cada archivo

| Archivo | Filas | Columnas | Tamaño | Periodo |
|---|---|---|---|---|
| `data/raw/caratula_ciiu.parquet` | 90.787 | 5 | 1,0 MB | Cortes 2016-2025 |
| `data/raw/caratula_atributos.parquet` | 607.190 | 5 | 4,9 MB | Cortes 2016-2025 |
| `data/raw/balance_largo.parquet` | 1.317.129 | 7 | 11,8 MB | Cortes 2016-2025 |
| `data/raw/resultados_largo.parquet` | 690.597 | 7 | 6,6 MB | Cortes 2016-2025 |
| `data/raw/insolvencia/*.xlsx` | 10 cortes (2.700 a 5.699 filas cada uno) | 14 a 19 | 4,5 MB | Dic. 2019 a oct. 2026 |
| `data/processed/panel_empresa_anio.parquet` | 79.651 | 67 | 24,8 MB | Años fiscales 2016-2025 |
| `data/processed/insolvencia_eventos.parquet` | 8.078 | 7 | 0,3 MB | Aperturas 1990-2026 |

El panel tiene 12.516 empresas (10.381 en el universo principal CIIU 41, 43 y 68). Los cortes de insolvencia
usados son 2019-12, 2020-03, 2020-06, 2020-09, 2021-12, 2022-12, 2023-12, 2024-12, 2025-12 y 2026-10.

## Calidad de los datos (resumen)

- Reportes por empresa-año antes de depurar: 1,03 a 1,04. Después de depurar queda un reporte por empresa-año.
- Sin total de activos reportado: 7,0 %. Sin deuda financiera reportada: 44,6 %. Con ingresos en cero: 10,6 %.
- La base de insolvencia trae años de apertura como entero, como fecha (corte 2020-03) y valores inválidos (por ejemplo `5`);
  se normalizan y se descartan los años fuera de 1990-2026.
- El cruce por NIT entre la base de insolvencia y los estados financieros es estable entre cortes: entre 57 % y 67 % de
  las empresas del sector (según el CIIU de la base de insolvencia) aparecen en los estados financieros.
- Solo el 38 % de los eventos del sector tiene estados financieros del año anterior.

## Licencia y fecha de corte

Estados financieros NIIF: Creative Commons Atribución-CompartirIgual 4.0 (datos.gov.co), actualización
de julio de 2026. Base de insolvencia: publicación de la Superintendencia de Sociedades; la fecha de cada
corte está en el nombre del archivo.
