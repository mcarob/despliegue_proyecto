import pandas as pd
import pytest

from alerta_insolvencia.config.core import config
from alerta_insolvencia.processing.data_manager import load_dataset


@pytest.fixture(scope="session")
def sample_input_data() -> pd.DataFrame:
    """Muestra de 2.000 empresa-año de la prueba 2022-2024 (incluye sus 66 eventos), sin identificadores."""
    return load_dataset(file_name=config.app_config.datos_prueba)


@pytest.fixture()
def registro_valido(sample_input_data) -> dict:
    """Un registro cualquiera de la muestra, como lo enviaría el tablero."""
    fila = sample_input_data.iloc[0]
    return {v: (None if pd.isna(fila[v]) else fila[v]) for v in config.modelo.variables}
