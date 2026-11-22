"""Configuración de la API: rutas, nombre, orígenes permitidos (CORS) y registro de mensajes (logs)."""

import logging
import sys
from types import FrameType
from typing import List, cast

from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Valores por defecto; cualquiera se puede cambiar con una variable de entorno del mismo nombre."""

    model_config = SettingsConfigDict(case_sensitive=True)

    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "API de alerta temprana de insolvencia"
    LOGGING_LEVEL: int = logging.INFO

    # Orígenes que pueden llamar la API desde un navegador. El tablero la llama desde su servidor
    # (no desde el navegador), así que no necesita estar en la lista.
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:8050",
        "http://127.0.0.1:8050",
    ]


class InterceptHandler(logging.Handler):
    """Envía a loguru los mensajes del logging estándar (los de uvicorn), para tener un solo formato."""

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = cast(FrameType, frame.f_back)
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_app_logging(config: Settings) -> None:
    """Prepara el registro de mensajes de la aplicación y de uvicorn."""
    logging.getLogger().handlers = [InterceptHandler()]
    for nombre in ("uvicorn.asgi", "uvicorn.access"):
        logging.getLogger(nombre).handlers = [InterceptHandler(level=config.LOGGING_LEVEL)]
    logger.configure(handlers=[{"sink": sys.stderr, "level": config.LOGGING_LEVEL}])


settings = Settings()
