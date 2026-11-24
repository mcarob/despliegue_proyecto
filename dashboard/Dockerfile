# Imagen del tablero (Dash). No incluye el modelo: las predicciones las pide a la API (variable API_URL)
FROM python:3.12-slim

# Usuario sin privilegios que ejecuta el tablero
RUN adduser --disabled-password --gecos '' dash-user

WORKDIR /opt/tablero

# Primero las dependencias: esta capa se reutiliza si solo cambia el código
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Código del tablero
COPY app.py datos.py modelo.py run.sh ./
COPY assets/ ./assets/
RUN mkdir -p artefactos datos && chmod +x run.sh && chown -R dash-user:dash-user ./

USER dash-user

# Dirección de la API y ruta de los datos (resumen_empresas, traído con dvc pull y montado como volumen)
ENV PORT=8050 \
    API_URL=http://localhost:8001 \
    DATOS_RESUMEN=/opt/tablero/datos/resumen_empresas.csv
EXPOSE 8050

CMD ["bash", "./run.sh"]
