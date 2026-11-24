# Despliegue de la API y el tablero

La API y el tablero se despliegan como dos contenedores Docker en una máquina EC2 de AWS (Ubuntu 24.04, t3.small,
disco de 20 GB), con el puerto 22 (SSH), el 8001 (API) y el 8050 (tablero) abiertos en el grupo de seguridad.

1. Instalar Docker y Git en la máquina.
2. Clonar el repositorio:
   `git clone https://github.com/mcarob/despliegue_proyecto.git`
3. Copiar el archivo de datos del tablero a `data/processed/resumen_empresas.parquet` (se trae con `dvc pull` o con `scp`).
4. Desde la raíz del repositorio, construir las imágenes y levantar los contenedores:
   `sudo docker compose up -d --build`
5. Comprobar que corren: `sudo docker compose ps` y `curl http://localhost:8001/api/v1/health`.
6. Abrir la API en `http://IP:8001/docs` y el tablero en `http://IP:8050` (el tablero tarda 1 o 2 minutos en abrir la
   primera vez).

El tablero llama a la API por la red interna de Docker (`API_URL=http://api:8001`). Las capturas del despliegue están en
`docs/soportes/despliegue/`.
