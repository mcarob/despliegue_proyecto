"""Esquemas de entrada y salida de POST /predict (FastAPI los usa para validar y para la documentación en /docs)."""

from typing import Any, List, Literal, Optional

from alerta_insolvencia.processing.validation import DataInputSchema
from pydantic import BaseModel, ConfigDict, Field


class ResultadoEmpresa(BaseModel):
    """Resultado del modelo para una empresa, en el mismo orden en que llegó."""

    puntaje: float = Field(..., description="Puntaje de riesgo de 0 a 1: sirve para ordenar a las empresas")
    probabilidad: float = Field(..., description="Probabilidad calibrada de abrir insolvencia el año siguiente")
    nivel_alerta: Literal["alto", "medio", "bajo"] = Field(
        ..., description="Alto: dentro del 10 % de mayor riesgo del año de referencia; medio: el 20 % siguiente"
    )


class PredictionResults(BaseModel):
    errors: Optional[Any] = None
    version: str
    predictions: Optional[List[float]] = Field(None, description="Puntajes de riesgo (lista simple, para el tablero)")
    resultados: Optional[List[ResultadoEmpresa]] = None


class MultipleDataInputs(BaseModel):
    """Una o varias empresas. Solo el subsector es obligatorio; los datos faltantes se imputan con la mediana."""

    inputs: List[DataInputSchema]

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "inputs": [
                {
                    "liquidez_corriente": 0.85, "endeudamiento": 0.93, "apalancamiento": 13.2,
                    "deuda_fin_activos": 0.41, "pasivo_corriente_pasivo": 0.78, "capital_trabajo_activos": -0.12,
                    "roa": -0.06, "roe": -0.85, "margen_neto": -0.21, "margen_operacional": -0.09,
                    "cobertura_intereses": -0.8, "rotacion_activos": 0.29, "efectivo_activos": 0.01,
                    "ganancias_acum_activos": -0.15, "ln_activos": 21.3, "altman_z2": -0.9,
                    "patrimonio_negativo": 0, "perdida_neta": 1, "ciiu_grupo": "41"
                },
                {
                    "liquidez_corriente": 2.4, "endeudamiento": 0.35, "apalancamiento": 0.54,
                    "deuda_fin_activos": 0.08, "pasivo_corriente_pasivo": 0.6, "capital_trabajo_activos": 0.31,
                    "roa": 0.07, "roe": 0.11, "margen_neto": 0.12, "margen_operacional": 0.16,
                    "cobertura_intereses": 9.5, "rotacion_activos": 0.6, "efectivo_activos": 0.12,
                    "ganancias_acum_activos": 0.25, "ln_activos": 22.1, "altman_z2": 5.8,
                    "patrimonio_negativo": 0, "perdida_neta": 0, "ciiu_grupo": "68"
                }
            ]
        }
    })
