from pydantic import BaseModel


class Health(BaseModel):
    """Estado de la API: nombre y versiones de la API y del modelo que está sirviendo."""

    name: str
    api_version: str
    model_version: str
