# Experimentos de modelos (Entrega 2)

Modelos de clasificación para anticipar, con un año de ventaja, qué constructoras e inmobiliarias abrirán un
proceso de insolvencia. Las corridas se registran en el servidor de MLflow del proyecto (EC2 de AWS), experimento
**`alerta-insolvencia`**.

## Archivos

| Archivo | Qué hace |
|---|---|
| `experiments/entrenar_modelos.py` | Entrena la línea base y los seis modelos, y registra en MLflow parámetros, métricas y el modelo de cada corrida |
| `notebooks/02_modelos.ipynb` | Exploración: eventos por año, comparación, curvas ROC, precisión-exhaustividad y de captura, importancia de variables y calibración |
| `docs/figuras/modelos_*.png` | Figuras del notebook para el reporte |
| `docs/soportes/mlflow_experimentos.png` | Soporte: las 7 corridas del experimento `alerta-insolvencia` en el servidor de MLflow de la EC2 (IP visible) |

## Cómo correrlo

```bash
pip install -r requirements.txt          # incluye mlflow==3.17.1, la misma versión del servidor
dvc pull                                 # o copiar resumen_empresas a data/processed/
export MLFLOW_TRACKING_URI=http://<IP-de-la-EC2>:8050
python experiments/entrenar_modelos.py --datos data/processed/resumen_empresas.csv
```

En Windows (PowerShell) la variable se define con `$env:MLFLOW_TRACKING_URI = "http://<IP>:8050"`. Para practicar sin
servidor se puede usar `sqlite:///mlflow.db` (el archivo y la carpeta `mlruns/` están en `.gitignore`).

## Decisiones

| Tema | Decisión | Por qué |
|---|---|---|
| Universo | CIIU 41, 43 y 68 | Construcción de edificios, actividades especializadas e inmobiliarias; las obras civiles (42) se dejan por fuera (comentario del tutor en la Entrega 1) |
| Objetivo | `y_h1`: apertura de insolvencia el año siguiente | Horizonte de alerta temprana de un año |
| Filas | Sin años posteriores al evento (`post_evento = 0`) y con etiqueta completa (`h1_valido = 1`) | Evita fuga de información y años sin respuesta conocida |
| Partición | Entrenamiento 2016-2021, prueba 2022-2024 | Validación fuera de tiempo: el modelo siempre predice años que no vio |
| Variables | 16 razones financieras, 2 indicadores (patrimonio negativo, pérdida neta) y el subsector | Las mismas de la exploración de la Entrega 1 |
| Preprocesamiento | Recorte a los percentiles 1 y 99 (calculados solo con entrenamiento), mediana para faltantes, *one-hot* del subsector; escalado solo para la logística | Las razones tienen colas extremas y faltantes cuando el denominador es cero |
| Desbalance | `class_weight="balanced"` | La prevalencia es 0,3 % a 0,8 % por año |
| Métricas | AUC, AUC-PR, AUC-PR sobre la prevalencia, captura de eventos en el 10 % de mayor riesgo y Brier | La exactitud no sirve con una clase tan rara; la captura en el 10 % es la lectura de negocio |

## Resultados en prueba (2022-2024: 21.881 empresa-año, 66 eventos, prevalencia 0,30 %)

| Corrida | AUC | AUC-PR | AUC-PR / prevalencia | Captura en el 10 % |
|---|:---:|:---:|:---:|:---:|
| `linea_base_altman` (Z'' sin entrenar) | 0,676 | 0,006 | 1,9 | 18,2 % |
| `logistica_C1` | 0,872 | 0,023 | 7,7 | 57,6 % |
| `logistica_C0.1` | 0,873 | 0,023 | 7,8 | 56,1 % |
| `random_forest_hoja20` | 0,886 | 0,048 | 15,9 | 65,2 % |
| **`random_forest_hoja50`** | **0,892** | 0,047 | 15,7 | **66,7 %** |
| `gradient_boosting_lr0.05` | 0,873 | 0,036 | 11,8 | 54,5 % |
| `gradient_boosting_lr0.1` | 0,873 | 0,029 | 9,5 | 62,1 % |

Entrenamiento: 35.810 empresa-año y 171 eventos (2016-2021).

**Lectura:** todos los modelos superan a la línea base de Altman. El random forest con hojas de al menos 50 empresas es
el mejor: revisando el 10 % de las empresas con mayor riesgo encuentra dos de cada tres que abren insolvencia al año
siguiente. Con solo 66 eventos de prueba las diferencias pequeñas entre modelos no son concluyentes. Las
probabilidades quedan infladas por `class_weight` (media de 0,08 en el random forest frente a una tasa real de
0,003): sirven para ordenar el riesgo; para mostrarlas como probabilidad hay que calibrarlas. Las variables con más
peso en el random forest son efectivo sobre activos, margen neto, endeudamiento y margen operacional.

---

# Iteración de la entrega final: ¿se puede mejorar el modelo?

Script: `experiments/iteracion_entrega_final.py` (experimento de MLflow **`alerta-insolvencia-iteracion`**, 27 corridas).
Resultados completos en `experiments/resultados_iteracion.json`; figuras `docs/figuras/modelos_04_calibracion.png` y
`docs/figuras/modelos_05_iteracion.png`.

**Qué se corrigió del método.** En la Entrega 2 los modelos se compararon directamente en la prueba 2022-2024, así que
la prueba dejó de ser independiente. Ahora la elección se hace con **validación temporal móvil**: para 2019, 2020 y
2021 se entrena con los años anteriores y se predice ese año (21.075 empresa-año, 94 eventos). Se decide con el AUC-PR
y la captura en el 10 % de la validación, y una alternativa se adopta solo si mejora de forma robusta (intervalo del
95 % por encima de cero). La prueba se reporta al final, como confirmación, y no cambia ninguna decisión.

| Qué se probó | Resultado en validación | Decisión |
|---|---|---|
| 8 modelos (random forest con hojas de 30, 50 y 100 o max_features 0,3; extra trees; gradient boosting regularizado con y sin balanceo; logística) | El random forest con hojas de 50 queda primero junto con el de hojas de 30 (AUC-PR 0,062; captura en el 10 % de 60 % frente a 59 %). Con el corte del 10 % dentro de cada año, como revisaría el banco, los mejores random forest quedan entre 55 % y 57 %, a un evento de distancia (de 94). Boosting, extra trees y logística quedan por debajo (AUC-PR de 0,026 a 0,045) | Se mantiene `random_forest_hoja50`: ninguna alternativa lo supera de forma robusta |
| 11 variables de tendencia (cambios frente al año anterior, crecimiento de activos e ingresos, edad, años reportados, pérdidas seguidas) y grupo NIIF | Suben un poco el AUC (0,826 a 0,840 en el mejor caso), pero bajan el AUC-PR y la captura en el 10 % | No se adoptan: más variables para la API sin ganancia en lo que usa el banco |
| Etiqueta ampliada: entrenar con "insolvencia en 1 o 2 años" (más eventos), **con purga** del año que filtraría el futuro | En las métricas con que se decide no hay mejora robusta: AUC-PR −0,006 (IC 95 % −0,026 a +0,011) y captura +3 puntos (IC −4 a +10). Solo sube el AUC (+0,024, IC +0,008 a +0,042) | No se adopta. La prueba lo confirma después: AUC −0,012 (IC −0,023 a −0,001) y captura −5 puntos |
| Calibración de Platt (ajustada con las predicciones fuera de tiempo 2019-2021) | Validación, calibrando cada año con los otros dos: Brier 0,0044 frente a 0,0044 de la tasa constante. Prueba: Brier 0,0171 → **0,0029** (tasa constante 0,0030); 92 eventos esperados frente a 66 observados | **Se adopta:** la probabilidad pasa de una media de 8 % a 0,4 %, el orden de la tasa real (0,3 %). Es una escala aproximada: la tasa del sector cambia de un año a otro (0,27 % a 0,76 % en 2019-2021) y la calibración no puede anticipar ese cambio |

**Lección sobre la fuga de información.** Sin la purga, la etiqueta ampliada parecía subir la captura de 60 % a 69 %:
la fila del año anterior quedaba marcada con un evento que todavía no se conocía al momento de predecir. Con la purga,
la ganancia en las métricas de decisión desaparece.

## Segunda ronda: búsqueda amplia

Script: `experiments/iteracion_ronda2.py` (experimento de MLflow **`alerta-insolvencia-ronda2`**). Resultados en
`experiments/resultados_ronda2.json`; figura `docs/figuras/modelos_06_ronda2.png`. Se usaron las mismas reglas de la
primera ronda, escritas en el script antes de correrlo. La comparación es pareada contra el modelo actual, y el
remuestreo se hace dentro de cada año.

| Qué se probó | Resultado en validación |
|---|---|
| **Ruido de fondo:** el modelo actual con 5 semillas más | El AUC-PR va de 0,054 a 0,062 y la captura por año de 56,4 % a 58,5 % **sin cambiar nada más que la semilla** |
| Random forest: 26 combinaciones (hojas de 5 a 100, variables por división, profundidad, tipo de balanceo, submuestreo) | Hojas más pequeñas (10 y 20) suben el AUC-PR a 0,068-0,070, pero no capturan más (55-57 %). Limitar la profundidad o quitar el balanceo empeora |
| Gradient boosting: 12 combinaciones (tasa, hojas, regularización, con y sin balanceo) | Todas por debajo: AUC-PR de 0,023 a 0,041 |
| Submuestreo balanceado en bolsas (todos los eventos y una muestra de no eventos en cada bolsa) | El mejor de la ronda en AUC-PR (0,072), con la misma captura (55 %) |
| Variables de historia (pérdidas seguidas, reportó el año anterior, edad, crecimiento de ingresos) | Empeoran: AUC-PR de 0,054 a 0,058 y captura de 52 % a 54 % |
| 5 combinaciones de modelos (promedio de rangos dentro de cada año) | La mejor llega a 58,5 % de captura, dentro del ruido de fondo |
| Estabilidad: el actual y el de hojas de 20 con 2.000 árboles y 3 semillas | Hojas de 20: AUC-PR de 0,066 a 0,071 frente a 0,057-0,063, pero captura de 56,4-57,4 % frente a 57,4-58,5 %. Ordena mejor los primeros puestos, pero no encuentra más eventos en el 10 % que revisa el banco |

**Decisión: se mantiene el modelo actual.** El mejor en validación fue el submuestreo con boosting. Frente al actual
mejora el AUC (+0,017, IC +0,005 a +0,031), pero no las métricas de decisión: AUC-PR +0,011 (IC −0,013 a +0,039) y
captura −0,7 puntos (IC −7 a +6). No cumple la regla.

**La prueba confirma la decisión.** Se miró una sola vez, al final. En 2022-2024 ese ganador queda por debajo del
actual: AUC-PR 0,042 frente a 0,047 y captura por año 65,2 % frente a 66,7 %. Es el efecto esperado al escoger el
mejor entre 49 alternativas con 94 eventos: parte de su ventaja en validación era azar. Por eso la regla exige un
intervalo por encima de cero, y no basta con quedar primero en la tabla.

**Lectura.** Con este tamaño de datos (171 eventos para entrenar y 94 para validar), las diferencias entre modelos
razonables son del mismo tamaño que cambiar la semilla. El límite está en los datos, no en el algoritmo. Lo que más
podría mejorar el modelo es tener más eventos (más años o más sectores), no un algoritmo más complejo.

**Modelo final (versión 2.0.0 del paquete, `model-pkg/`).** Es el random forest de la Entrega 2 más la calibración, con
niveles de alerta: alto si está en el 10 % de mayor puntaje de 2025, medio si está en el 20 % siguiente.

| Prueba 2022-2024 (66 eventos) | Modelo | IC 95 % | Z'' de Altman | Endeudamiento |
|---|:---:|:---:|:---:|:---:|
| AUC | **0,892** | 0,850-0,927 | 0,676 | 0,755 |
| AUC-PR | **0,047** (15,7 veces la prevalencia) | 0,025-0,107 | 0,006 | 0,008 |
| Captura al revisar el 5 %, 10 % y 20 % | 43,9 %, **66,7 %**, 83,3 % | 56,2 %-78,0 % (10 %) | 10,6 %, 18,2 %, 36,4 % | 12,1 %, 27,3 %, 53,0 % |
| Precisión de la lista del 10 % | 2,0 % (6,7 veces la tasa) | | 0,5 % | 0,8 % |

La validación móvil da un AUC más bajo que la prueba (0,826 frente a 0,892; por año: 0,816 en 2019, 0,818 en 2020 y
0,849 en 2021). Los primeros pliegues entrenan con solo 3 o 4 años, y el AUC sube a medida que hay más historia. Con
pocas decenas de eventos por periodo, el valor puntual varía bastante; por eso se reporta el intervalo de confianza.

**Cómo reproducirlo**

```bash
export MLFLOW_TRACKING_URI=http://<IP-de-la-EC2>:8050     # o sqlite:///mlflow.db sin servidor
python experiments/iteracion_entrega_final.py --datos data/processed/resumen_empresas.csv   # unos 5 minutos
cd model-pkg && tox -e train       # entrena el paquete y registra la corrida paquete_v2.0.0_rf_calibrado
```
