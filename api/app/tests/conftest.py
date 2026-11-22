from typing import Generator

import pandas as pd
import pytest
from alerta_insolvencia.config.core import config
from alerta_insolvencia.processing.data_manager import load_dataset
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def test_data() -> pd.DataFrame:
    """Muestra de 2.000 empresa-año que viene dentro del paquete del modelo."""
    return load_dataset(file_name=config.app_config.datos_prueba)


@pytest.fixture()
def client() -> Generator:
    with TestClient(app) as _client:
        yield _client
        app.dependency_overrides = {}
