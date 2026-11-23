"""Validación de las entradas del modelo (la misma que usa la API para documentar y revisar cada campo)."""

import json
import typing as t

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from alerta_insolvencia.config.core import config

MC = config.modelo

# Un número real finito (sin infinitos ni NaN explícitos); None significa dato faltante y se imputa con la mediana
Razon = t.Annotated[t.Optional[float], Field(default=None, allow_inf_nan=False)]
Indicador = t.Annotated[t.Optional[int], Field(default=None, ge=0, le=1)]


class DataInputSchema(BaseModel):
    """Razones financieras de una empresa en un año (estados financieros NIIF reportados a Supersociedades)."""

    model_config = ConfigDict(extra="ignore")

    liquidez_corriente: Razon = Field(None, description="Activo corriente / pasivo corriente")
    endeudamiento: Razon = Field(None, description="Pasivo total / activo total")
    apalancamiento: Razon = Field(None, description="Pasivo total / patrimonio")
    deuda_fin_activos: Razon = Field(None, description="Deuda financiera / activo total")
    pasivo_corriente_pasivo: Razon = Field(None, description="Pasivo corriente / pasivo total")
    capital_trabajo_activos: Razon = Field(None, description="(Activo corriente - pasivo corriente) / activo total")
    roa: Razon = Field(None, description="Utilidad neta / activo total")
    roe: Razon = Field(None, description="Utilidad neta / patrimonio")
    margen_neto: Razon = Field(None, description="Utilidad neta / ingresos")
    margen_operacional: Razon = Field(None, description="Utilidad operacional / ingresos")
    cobertura_intereses: Razon = Field(None, description="Utilidad operacional / costos financieros")
    rotacion_activos: Razon = Field(None, description="Ingresos / activo total")
    efectivo_activos: Razon = Field(None, description="Efectivo / activo total")
    ganancias_acum_activos: Razon = Field(None, description="Ganancias acumuladas / activo total")
    ln_activos: Razon = Field(None, description="Logaritmo natural del activo total (tamaño)")
    altman_z2: Razon = Field(None, description="Z'' de Altman para empresas no manufactureras")
    patrimonio_negativo: Indicador = Field(None, description="1 si el patrimonio es negativo")
    perdida_neta: Indicador = Field(None, description="1 si la utilidad neta del año es negativa")
    ciiu_grupo: t.Literal["41", "43", "68"] = Field(
        ..., description="Subsector CIIU: 41 construcción de edificios, 43 actividades especializadas, 68 inmobiliarias"
    )

    @field_validator("ciiu_grupo", mode="before")
    @classmethod
    def dos_digitos(cls, valor: t.Any) -> t.Any:
        """Acepta 41, "41" o "4111" (código completo) y deja los dos primeros dígitos."""
        return str(valor).strip()[:2] if valor is not None else valor


class MultipleDataInputs(BaseModel):
    inputs: t.List[DataInputSchema]


def validate_inputs(*, input_data: pd.DataFrame) -> t.Tuple[pd.DataFrame, t.Optional[list]]:
    """Revisa cada registro y devuelve los datos listos para el modelo y la lista de errores (o None).

    Las columnas que no vengan se toman como faltantes; las que sobren se ignoran.
    """
    repetidas = sorted(set(input_data.columns[input_data.columns.duplicated()].astype(str)))
    if repetidas:
        # Con dos columnas del mismo nombre no se sabe cuál usar: se rechaza en vez de escoger una al azar
        return input_data, [{"type": "columnas_repetidas", "loc": ["inputs"],
                             "msg": f"Columnas repetidas en la entrada: {', '.join(repetidas)}."}]
    datos = input_data.reindex(columns=MC.variables)
    errores = None
    try:
        registros = datos.astype(object).replace({np.nan: None}).to_dict(orient="records")
        # Los faltantes se quitan del registro: así una razón ausente toma su valor por defecto (se imputa) y un
        # subsector ausente se informa como "campo obligatorio", igual que en la API
        registros = [{k: v for k, v in r.items() if v is not None} for r in registros]
        validados = MultipleDataInputs(inputs=registros)
        datos = pd.DataFrame([r.model_dump() for r in validados.inputs]).reindex(columns=MC.variables)
        datos[MC.razones + MC.indicadores] = datos[MC.razones + MC.indicadores].astype(float)
    except ValidationError as error:
        errores = json.loads(error.json(include_url=False))
    return datos, errores
