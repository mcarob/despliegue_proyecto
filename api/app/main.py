"""Punto de entrada de la API: crea la aplicación FastAPI, la página de inicio y las rutas de la versión 1.

Uso local (desde la carpeta api/):
    uvicorn app.main:app --host 0.0.0.0 --port 8001
y abrir http://localhost:8001/docs para probar los servicios.
"""

import math
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger

from app.api import api_router
from app.config import settings, setup_app_logging

# El registro de mensajes se configura antes de crear la aplicación para no perder los primeros
setup_app_logging(config=settings)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "Sirve el modelo que anticipa, con un año de ventaja, qué constructoras e inmobiliarias abrirán un "
        "proceso de insolvencia. Recibe las razones financieras de una o varias empresas y devuelve su puntaje "
        "de riesgo, la probabilidad calibrada y el nivel de alerta."
    ),
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)


def _sin_no_finitos(valor: float) -> Any:
    """JSON no admite NaN ni infinitos: dentro del mensaje de error se muestran como texto."""
    return valor if math.isfinite(valor) else str(valor)


@app.exception_handler(RequestValidationError)
async def error_de_validacion(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Responde 422 con el detalle del campo que falló.

    El manejador por defecto copia el valor recibido en la respuesta. Si llega un NaN o un infinito, ese valor no se
    puede escribir como JSON y la API terminaría en un error 500 en vez de explicar qué campo está mal.
    """
    detalle = jsonable_encoder(exc.errors(), custom_encoder={float: _sin_no_finitos})
    return JSONResponse(status_code=422, content={"detail": detalle})


root_router = APIRouter()


@root_router.get("/", response_class=HTMLResponse)
def index() -> Any:
    """Página de inicio con el enlace a la documentación interactiva."""
    body = (
        "<html><body style='font-family: Arial; padding: 16px;'>"
        f"<h1>{settings.PROJECT_NAME}</h1>"
        "<p>Modelo de alerta temprana de insolvencia de constructoras e inmobiliarias (MIAD, Uniandes).</p>"
        "<p>Documentación y pruebas: <a href='/docs'>/docs</a> · "
        f"Estado: <a href='{settings.API_V1_STR}/health'>{settings.API_V1_STR}/health</a></p>"
        "</body></html>"
    )
    return HTMLResponse(content=body)


app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(root_router)

if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


if __name__ == "__main__":
    # Solo para desarrollo; en el contenedor la API se inicia con uvicorn desde run.sh
    logger.warning("Ejecutando en modo de desarrollo.")
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")
