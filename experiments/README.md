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
