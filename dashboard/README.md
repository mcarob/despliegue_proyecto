# Tablero: alerta temprana de insolvencia (entrega final)

Tablero en Dash con las cinco vistas de la maqueta (`mockup/`): panorama del sector, lista de vigilancia, ficha de
empresa, simulador y salud del modelo. **No incluye el modelo:** todas las predicciones las pide a la API de
inferencia (`api/`), que sirve el paquete `alerta_insolvencia` 2.0.0.

## Cómo se conecta con la API

| Uso en el tablero | Servicio de la API |
|---|---|
| Al iniciar: puntaje, probabilidad calibrada y nivel de alerta de todas las empresas-año (en lotes de 5.000) | `POST /api/v1/predict` |
| Simulador y explicación del puntaje en la ficha | `POST /api/v1/predict` |
| Vista de salud: métricas, intervalos, curva de captura, calibración e importancia de variables | `GET /api/v1/modelo` |
| Esperar a que la API esté lista y saber qué versión del modelo sirve | `GET /api/v1/health` |

La dirección de la API se indica con la variable de entorno `API_URL` (por defecto `http://localhost:8001`).
Los puntajes se guardan en `dashboard/artefactos/` por versión del modelo: si la API pasa a servir otra versión,
el tablero los vuelve a pedir.

## Cómo correrlo

**Con Docker (como en la nube).** Con la API ya encendida y el archivo `resumen_empresas.csv` en una carpeta de la
máquina (se obtiene con `dvc pull`):

```bash
cd dashboard
docker build -t alerta-insolvencia-tablero:1.0.0 .
docker run -d --name tablero -p 8050:8050 \
  -e API_URL=http://<IP-de-la-API>:8001 \
  -v /ruta/a/data/processed:/opt/tablero/datos:ro \
  alerta-insolvencia-tablero:1.0.0
```

**Sin Docker (desarrollo),** desde la raíz del repositorio y con la API encendida:

```bash
pip install -r dashboard/requirements.txt
dvc pull                                   # o copiar resumen_empresas en data/processed/
export API_URL=http://localhost:8001       # en Windows: set API_URL=http://localhost:8001
python dashboard/app.py
```

Abrir http://127.0.0.1:8050 (o la IP pública de la máquina y el puerto 8050).

## Archivos

| Archivo | Qué hace |
|---|---|
| `app.py` | Estructura del tablero, filtros globales y las cinco vistas |
| `modelo.py` | Cliente de la API: envía las empresas a `/predict`, trae la ficha de `/modelo` y prepara los puntajes |
| `datos.py` | Lee `resumen_empresas` (ruta configurable con `DATOS_RESUMEN`), deja el universo CIIU 41, 43 y 68 y anonimiza el NIT |
| `assets/estilo.css` | Estilos (colores y distribución de la maqueta) |
| `Dockerfile`, `run.sh` | Imagen del tablero; se ejecuta con gunicorn en el puerto 8050 |

## Decisiones

- **Universo:** CIIU 41, 43 y 68; las obras civiles (42) quedan por fuera.
- **Puntaje y probabilidad:** el puntaje (0-100) ordena el riesgo; la probabilidad calibrada (Platt) estima la
  chance de abrir insolvencia el año siguiente, en el orden de la tasa real del sector (0,3 %).
- **Bandas:** alta = el porcentaje de la capacidad de revisión con mayor puntaje; media = el 20 % siguiente.
- **Sin reporte reciente:** las empresas que no reportaron en el año elegido se califican con su último reporte
  de los dos años anteriores y se marcan, porque no reportar es una señal de riesgo.
- **Qué explica el puntaje:** cuánto cambia el puntaje (calculado por la API) si cada variable toma la mediana de
  su subsector.
