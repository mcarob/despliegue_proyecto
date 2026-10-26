# Recursos de datos y de dónde se descargaron

Fecha de descarga de todos los recursos: 25 de octubre de 2026. Los tres conjuntos NIIF son de la Superintendencia de Sociedades (licencia Creative Commons Atribución-CompartirIgual 4.0) y se obtienen de datos.gov.co.

## Estados financieros NIIF (datos.gov.co)

| # | Recurso | Página del recurso | Cómo se descargó |
|---|---|---|---|
| 1 | Estados Financieros NIIF: Estado de Situación Financiera (`pfdp-zks5`) | https://www.datos.gov.co/d/pfdp-zks5 | API SODA `https://www.datos.gov.co/resource/pfdp-zks5.json`, filtrada por NIT del sector, con `src/extract_niif.py` |
| 2 | Estados Financieros NIIF: Estado de Resultado Integral (`prwj-nzxa`) | https://www.datos.gov.co/d/prwj-nzxa | API SODA `https://www.datos.gov.co/resource/prwj-nzxa.json`, con `src/extract_niif.py` |
| 3 | Estados Financieros NIIF: Carátula (`6hqw-m3dm`) | https://www.datos.gov.co/d/6hqw-m3dm | API SODA `https://www.datos.gov.co/resource/6hqw-m3dm.json`, con `src/extract_niif.py` |

## Base de datos de insolvencia (Superintendencia de Sociedades)

Página donde se publican todos los cortes: https://www.supersociedades.gov.co/es/web/procedimientos-de-insolvencia/informes (sección Base de Datos Insolvencia).

| # | Recurso (corte) | Archivo original | Enlace de descarga | Cómo se descargó |
|---|---|---|---|---|
| 4 | Base de insolvencia, corte 2019-12 | `BASE_SOPORTE_INSOLVENCIA_DIC_2019.xlsx` | https://www.supersociedades.gov.co/documents/58444/6516407/BASE_SOPORTE_INSOLVENCIA_DIC_2019.xlsx/4b4c0c48-4f23-3cc6-35df-df2b329679fb?version=1.0&t=1694034011022 | Script, petición HTTP directa |
| 5 | Base de insolvencia, corte 2020-03 | `BASE_SOPORTE_INSOLVENCIA_MAR_2020.xlsx` | https://www.supersociedades.gov.co/documents/58444/6516168/BASE_SOPORTE_INSOLVENCIA_MAR_2020.xlsx/2353b53d-0b4a-4317-baba-98c57eb3a542?version=1.0&t=1694033361992 | Script, petición HTTP directa |
| 6 | Base de insolvencia, corte 2020-06 | `BASE_SOPORTE_INSOLVENCIA_JUN_2020.xlsx` | https://www.supersociedades.gov.co/documents/58444/6516168/BASE_SOPORTE_INSOLVENCIA_JUN_2020.xlsx/d9b8d5ec-7d23-7579-7b6a-13023958b62f?version=1.0&t=1694033472440 | Script, petición HTTP directa |
| 7 | Base de insolvencia, corte 2020-09 | `BASE_SOPORTE_INSOLVENCIA_SEP_2020.xlsx` | https://www.supersociedades.gov.co/documents/58444/6516168/BASE_SOPORTE_INSOLVENCIA_SEP_2020.xlsx/ffeb2f14-5607-73c6-8591-ef2ee9f1fc94?version=1.0&t=1694033963280 | Manual, desde el navegador |
| 8 | Base de insolvencia, corte 2021-12 | `Base-datos-insolvencia-diciembre-2021.xlsx` | https://www.supersociedades.gov.co/documents/58444/6516010/Base-datos-insolvencia-diciembre-2021.xlsx/ffbd0733-6430-c60e-d454-6a47367b6ecd?version=1.0&t=1694028376290 | Manual, desde el navegador |
| 9 | Base de insolvencia, corte 2022-12 | `BASE-SOPORTE-INSOLVENCIA-2022-CORTE-DICIEMBRE.xlsx` | https://www.supersociedades.gov.co/documents/58444/6515849/BASE-SOPORTE-INSOLVENCIA-2022-CORTE-DICIEMBRE.xlsx/193a9c81-3986-ed94-af98-cdc8ac6c68a9?version=1.0&t=1694026647657 | Manual, desde el navegador |
| 10 | Base de insolvencia, corte 2023-12 | `BASE-SOPORTE-INSOLVENCIA-2023-CORTE-DICIEMBRE.xlsx` | https://www.supersociedades.gov.co/documents/58444/6515839/BASE-SOPORTE-INSOLVENCIA-2023-CORTE-DICIEMBRE.xlsx/4e32b9f3-5c45-b1a8-8606-a431fb3f21a3?version=1.1&t=1724249141759 | Manual, desde el navegador |
| 11 | Base de insolvencia, corte 2024-12 | `Base-Soporte-Insolvencia-2024-CORTE-DICIEMBRE.xlsx` | https://www.supersociedades.gov.co/documents/58444/7803983/Base-Soporte-Insolvencia-2024-CORTE-DICIEMBRE.xlsx/a5476d0a-46fe-c53f-ea36-417c206c1838?version=1.1&t=1735911388000 | Manual, desde el navegador |
| 12 | Base de insolvencia, corte 2025-12 | `Base-Soporte-Insolvencia-2025-CORTE-DICIEMBRE.xlsx` | https://www.supersociedades.gov.co/documents/58444/8332056/Base-Soporte-Insolvencia-2025-CORTE-DICIEMBRE.xlsx/b3b136e2-234a-d640-c79e-5ab0af1fd4c9?version=1.0&t=1767619414849 | Manual, desde el navegador |
| 13 | Base de insolvencia, corte 2026-10 | `Base-Soporte-Insolvencia-2026-CORTE-OCTUBRE.xlsx` | https://www.supersociedades.gov.co/documents/58444/9506465/Base-Soporte-Insolvencia-2026-CORTE-OCTUBRE.xlsx/3505ebea-c49e-c42f-00ea-48b443a1bc39?version=1.0&t=1791380389241 | Manual, desde el navegador |

Los archivos se guardan como `data/raw/insolvencia/insolvencia_AAAA-MM.xlsx`. Las URLs de los diez cortes están también en `src/insolvencia_urls.json`.
