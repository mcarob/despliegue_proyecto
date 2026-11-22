# Imagen de la API de inferencia (FastAPI + paquete del modelo)
FROM python:3.12-slim

# Usuario sin privilegios que ejecuta la API
RUN adduser --disabled-password --gecos '' api-user

WORKDIR /opt/api

# Primero las dependencias (incluye el .whl del modelo): esta capa se reutiliza si solo cambia el código
COPY requirements.txt ./
COPY model-pkg/ ./model-pkg/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Código de la API
COPY app/ ./app/
COPY run.sh ./
RUN chmod +x run.sh && chown -R api-user:api-user ./

USER api-user

# Puerto de la API (se puede cambiar con la variable PORT)
ENV PORT=8001
EXPOSE 8001

# Verificación de salud que usa Docker para saber si el contenedor está sano
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", 8001)}/api/v1/health')" || exit 1

CMD ["bash", "./run.sh"]
