# API de inferencia: alerta temprana de insolvencia

API en FastAPI que sirve el modelo empaquetado (`alerta_insolvencia` 2.0.0, ver `model-pkg/README.md`). El tablero
le envía las razones financieras de una o varias empresas y recibe el puntaje de riesgo, la probabilidad calibrada y
el nivel de alerta de cada una.

## Servicios

| Método y ruta | Qué hace |
|---|---|
| `GET /` | Página de inicio con enlaces a la documentación |
| `GET /docs` | Documentación interactiva (Swagger): permite probar cada servicio desde el navegador |
| `GET /api/v1/health` | Estado: nombre, versión de la API y versión del modelo |
| `POST /api/v1/predict` | Predicción para 1 a 5.000 empresas por llamada |
| `GET /api/v1/modelo` | Ficha del modelo: periodos, métricas de validación y prueba, calibración, umbrales e importancia de variables |

**Entrada de `/predict`.** Las 16 razones, los 2 indicadores y el subsector de cada empresa. Solo `ciiu_grupo`
(`"41"`, `"43"` o `"68"`) es obligatorio; un dato faltante (`null`) se imputa con la mediana de entrenamiento.

```json
{"inputs": [{"liquidez_corriente": 0.85, "endeudamiento": 0.93, "deuda_fin_activos": 0.41, "roa": -0.06,
             "margen_operacional": -0.09, "efectivo_activos": 0.01, "altman_z2": -0.9,
             "patrimonio_negativo": 0, "perdida_neta": 1, "ciiu_grupo": "41"}]}
```

**Salida.** `predictions` es la lista simple de puntajes, para ordenar la lista de vigilancia. `resultados` trae el
detalle de cada empresa, en el mismo orden de la entrada:

```json
{"errors": null, "version": "2.0.0", "predictions": [0.3644],
 "resultados": [{"puntaje": 0.3644, "probabilidad": 0.0264, "nivel_alerta": "alto"}]}
```

Códigos de respuesta:
- **422:** un campo no cumple el esquema: subsector ausente o fuera del universo, texto que no es un número en una
  razón, `NaN` o infinito (JSON no los admite; un dato faltante va como `null`), indicador distinto de 0 y 1. La
  respuesta dice qué campo de qué empresa falló.
- **400:** el lote está vacío o pasa de 5.000 registros.

## Correr la API

**Con Docker (como en la nube):**

```bash
cd api
docker build -t alerta-insolvencia-api:2.0.0 .
docker run -d --name api -p 8001:8001 alerta-insolvencia-api:2.0.0
curl http://localhost:8001/api/v1/health
```

El contenedor corre con un usuario sin privilegios, expone el puerto 8001 (variable `PORT`) y trae una verificación
de salud: `docker ps` lo muestra como `healthy` cuando responde. Con `docker stop` se cierra limpiamente (run.sh
lanza uvicorn con `exec`). La imagen pesa unos 720 MB.

**Sin Docker (desarrollo):**

```bash
cd api
pip install -r requirements.txt        # instala el .whl del modelo desde model-pkg/
uvicorn app.main:app --port 8001       # o: tox -e run
```

**Pruebas:**
- `tox -e test_app`: 8 pruebas sobre health, predicción, ejemplo de la documentación, errores 422 (incluidos `NaN` e
  infinitos) y 400, ficha del modelo y página de inicio.
- `tox -e checks`: revisa el estilo.

## Cómo la usa el tablero

El tablero llama la API desde su servidor. La URL se pasa con una variable de entorno; en Docker sería, por ejemplo,
`API_URL=http://<IP-de-la-EC2>:8001`:

```python
import os, requests

API_URL = os.environ.get("API_URL", "http://localhost:8001")

def calcular_riesgo(registros):
    respuesta = requests.post(f"{API_URL}/api/v1/predict", json={"inputs": registros}, timeout=30)
    respuesta.raise_for_status()
    return np.array(respuesta.json()["predictions"])
```

Los registros deben llevar `None` en lugar de `NaN`, porque JSON no admite `NaN`. Los puntajes son los mismos del
modelo v1.0 del tablero (misma configuración y semilla), así que la lista de vigilancia no cambia al conectarlo.

## Archivos

| Archivo | Qué hace |
|---|---|
| `app/main.py` | Crea la aplicación, la página de inicio y el CORS |
| `app/api.py` | Rutas `/health`, `/predict` y `/modelo` |
| `app/schemas/` | Esquemas de entrada y salida; la entrada reutiliza la validación del paquete del modelo |
| `app/config.py` | Nombre, prefijo `/api/v1`, orígenes permitidos y registro de mensajes |
| `model-pkg/alerta_insolvencia-2.0.0-py3-none-any.whl` | Paquete del modelo que se instala en la imagen |
| `Dockerfile`, `run.sh` | Imagen de la API e inicio con uvicorn |
| `tox.ini`, `requirements.txt` | Ambientes de prueba y dependencias |

Para actualizar el modelo, se reentrena y se construye el paquete en `model-pkg/` (`tox -e train` y `tox -e build`), se
copia el nuevo `.whl` en `api/model-pkg/`, se cambia su nombre en `requirements.txt` y se reconstruye la imagen.
