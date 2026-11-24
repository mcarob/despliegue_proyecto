#!/bin/bash
# Inicia el tablero con gunicorn en el puerto PORT (8050 por defecto).
# gthread con varios hilos evita que una conexión inactiva del navegador bloquee al trabajador (WORKER TIMEOUT);
# el timeout largo da tiempo a la primera carga, en la que el tablero pide a la API los puntajes de todas las empresas.
exec gunicorn app:server --bind 0.0.0.0:"${PORT:-8050}" --workers 1 --worker-class gthread --threads 4 --timeout 600
