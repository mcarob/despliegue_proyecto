"""Paquete del modelo de alerta temprana de insolvencia de constructoras e inmobiliarias.

Uso típico:
    from alerta_insolvencia.predict import make_prediction
    resultado = make_prediction(input_data=[{"endeudamiento": 0.9, ..., "ciiu_grupo": "41"}])
"""

from pathlib import Path

# La versión del paquete es también la versión del modelo entrenado que trae dentro
with open(Path(__file__).resolve().parent / "VERSION") as archivo_version:
    __version__ = archivo_version.read().strip()
