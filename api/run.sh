#!/bin/bash
# Inicia la API en el puerto PORT (8001 por defecto) escuchando en todas las interfaces del contenedor
uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8001}"
