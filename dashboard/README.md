# Tablero: alerta temprana de insolvencia (Entrega 2)

Tablero en Dash con las cinco vistas de la maqueta (`mockup/`): panorama del sector, lista de vigilancia,
ficha de empresa, simulador y salud del modelo. Usa datos reales de `resumen_empresas` y el modelo ganador
de la Entrega 2 (`random_forest_hoja50`, ver `experiments/README.md`).

## Cómo correrlo

Desde la raíz del repositorio:

```bash
pip install -r dashboard/requirements.txt
dvc pull                      # o copiar resumen_empresas (.parquet o .csv) en data/processed/
python dashboard/app.py
```

Abrir http://127.0.0.1:8050 en el navegador. La primera vez entrena el modelo (menos de un minuto) y guarda
el modelo, los puntajes y las métricas en `dashboard/artefactos/` (carpeta fuera de Git).

## Archivos

| Archivo | Qué hace |
|---|---|
| `app.py` | Estructura del tablero, filtros globales y las cinco vistas |
| `datos.py` | Lee `resumen_empresas`, deja el universo CIIU 41, 43 y 68 y anonimiza el NIT |
| `modelo.py` | Entrena el random forest con la misma configuración de `experiments/entrenar_modelos.py`, calcula puntajes y métricas; `calcular_riesgo()` es el punto que en la entrega final llamará a la API |
| `transformaciones.py` | Recorte de valores extremos usado por el modelo |
| `assets/estilo.css` | Estilos (colores y distribución de la maqueta) |

## Decisiones

- **Universo:** CIIU 41, 43 y 68; las obras civiles (42) quedan por fuera, como pidió el tutor en la Entrega 1.
- **Modelo:** se reentrena localmente con la misma configuración y semilla de la corrida registrada en MLflow
  (AUC 0,892 y 66,7 % de eventos capturados en el 10 % en la prueba 2022-2024), para no depender de que la EC2
  esté encendida. En la entrega final el simulador consumirá la API (`POST /predict`).
- **Puntaje y no probabilidad:** el balanceo de clases infla las probabilidades; el tablero muestra un puntaje
  de 0 a 100, el puesto y la banda de riesgo.
- **Bandas:** alta = el porcentaje de la capacidad de revisión con mayor puntaje; media = el 20 % siguiente.
- **Sin reporte reciente:** las empresas que no reportaron en el año elegido se califican con su último reporte
  de los dos años anteriores y se marcan, porque no reportar es una señal de riesgo.
- **Qué explica el puntaje:** cuánto cambia el puntaje si cada variable toma la mediana de su subsector.
