# Paquete del modelo: `alerta-insolvencia` 2.0.0

Modelo empaquetado que anticipa, con un año de ventaja, qué constructoras e inmobiliarias (CIIU 41, 43 y 68)
abrirán un proceso de insolvencia. El `.whl` trae dentro el modelo entrenado, sus metadatos (métricas,
calibración, umbrales de alerta e importancia de variables), la configuración y una muestra para las pruebas.
La API (`api/`) lo instala para servir las predicciones.

## Qué devuelve

```python
from alerta_insolvencia.predict import make_prediction

make_prediction(input_data=[{"endeudamiento": 0.93, "roa": -0.06, "altman_z2": -0.9, "perdida_neta": 1,
                             "ciiu_grupo": "41"}])
# {"predictions": [0.282], "probabilidades": [0.017], "niveles_alerta": ["alto"], "version": "2.0.0", "errors": None}
```

| Salida | Qué es | Para qué sirve |
|---|---|---|
| `predictions` | Puntaje de riesgo de 0 a 1 del random forest | Ordenar a las empresas (lista de vigilancia del tablero) |
| `probabilidades` | Probabilidad calibrada (Platt) de abrir insolvencia el año siguiente | Leer el riesgo en la escala real (la tasa del sector es 0,3 %) |
| `niveles_alerta` | `alto`, `medio` o `bajo` | Alto = dentro del 10 % de mayor puntaje de 2025; medio = el 20 % siguiente |

Solo `ciiu_grupo` es obligatorio. Las razones que falten se imputan con la mediana de entrenamiento; un subsector
fuera del universo, un texto en una razón, un infinito o un indicador distinto de 0 y 1 se rechazan con el detalle
del error.

## Modelo y resultados

- **Modelo:** random forest de 500 árboles con hojas de al menos 50 empresas y balanceo de clases (la corrida
  `random_forest_hoja50` de la Entrega 2), con 19 variables: 16 razones financieras, 2 indicadores y el subsector.
- **Elección:** en la iteración de la entrega final (`experiments/iteracion_entrega_final.py`) se compararon
  24 combinaciones de variables y modelos con validación temporal móvil (2019, 2020, 2021), sin tocar la prueba.
  Este modelo quedó entre los mejores, y ninguna alternativa lo superó de forma robusta.
- **Calibración:** las probabilidades del bosque están infladas por el balanceo (media 0,08 frente a una tasa de
  0,003). La calibración de Platt, ajustada con las predicciones fuera de tiempo de 2019-2021, las lleva a la
  escala real.

| Prueba fuera de tiempo 2022-2024 (21.881 empresa-año, 66 eventos) | Valor | IC 95 % |
|---|:---:|:---:|
| AUC | 0,892 | 0,850-0,927 |
| AUC-PR (15,7 veces la prevalencia de 0,30 %) | 0,047 | 0,025-0,107 |
| Eventos capturados al revisar el 10 % de mayor riesgo | 66,7 % | 56,2 %-78,0 % |
| Brier sin calibrar, calibrado y con la tasa constante | 0,0171; **0,0029**; 0,0030 | |

Líneas base en la misma prueba: Z'' de Altman (AUC 0,676; captura 18,2 %) y endeudamiento (AUC 0,755; captura 27,3 %).
Los valores exactos están en `alerta_insolvencia/trained/metadatos-modelo-v2.0.0.json`.

## Estructura

```
model-pkg/
  alerta_insolvencia/
    config.yml                 variables, periodos, hiperparámetros y umbrales (todo se cambia aquí)
    config/core.py             lee y valida config.yml
    processing/
      data_manager.py          lectura de datos, partición temporal, guardar y cargar el modelo
      transformaciones.py      recorte de valores extremos (percentiles 1 y 99)
      validation.py            esquema de entrada (lo reutiliza la API)
      calibracion.py           calibración de Platt y niveles de alerta
    pipeline.py                preprocesamiento + random forest
    train_pipeline.py          validación móvil, entrenamiento, prueba, calibración y registro en MLflow
    predict.py                 make_prediction()
    evaluacion.py              métricas para clase rara (AUC, AUC-PR, captura, Brier, bootstrap)
    trained/                   modelo entrenado y metadatos de la versión
    datasets/muestra_prueba.csv  2.000 empresa-año de 2022-2024 sin identificadores (para las pruebas)
  tests/                       13 pruebas: predicción, calidad mínima, calibración, niveles y validación
  tox.ini                      train, test_package, build y checks
```

## Entrenar, probar y construir

Requisitos: Python 3.12 y `pip install tox`. Los datos se traen desde la raíz del repositorio con `dvc pull`
(o se indica el archivo con la variable `ALERTA_DATOS`).

```bash
cd model-pkg
tox -e train          # entrena (unos 20 s), guarda el modelo en trained/ y corre las pruebas
tox -e test_package   # solo las pruebas, con el modelo que ya trae el paquete
tox -e build          # construye dist/alerta_insolvencia-2.0.0-py3-none-any.whl
cp dist/alerta_insolvencia-2.0.0-py3-none-any.whl ../api/model-pkg/   # lo que instala la API
```

Para registrar el entrenamiento en MLflow se define antes `MLFLOW_TRACKING_URI` (en PowerShell:
`$env:MLFLOW_TRACKING_URI = "http://<IP-de-la-EC2>:8050"`). La corrida queda como `paquete_v2.0.0_rf_calibrado`
en el experimento `alerta-insolvencia`.

**Versiones:** el modelo se guarda con scikit-learn 1.9.1 y por eso la dependencia está fijada en esa versión.
Abrirlo con otra versión puede fallar o cambiar las predicciones.
