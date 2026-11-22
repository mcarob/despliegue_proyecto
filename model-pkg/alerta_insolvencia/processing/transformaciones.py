"""Transformación propia del pipeline: recorte de valores extremos.

Vive en el paquete (y no en un script) para que el modelo guardado con joblib se pueda abrir desde la API,
el tablero o cualquier otro programa que tenga el paquete instalado.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RecorteExtremos(BaseEstimator, TransformerMixin):
    """Recorta cada variable a sus percentiles 1 y 99 calculados en entrenamiento.

    Las razones financieras tienen colas enormes (un denominador cercano a cero produce valores de miles),
    y sin el recorte unos pocos casos dominarían los cortes de los árboles.
    """

    def __init__(self, inferior: float = 0.01, superior: float = 0.99):
        self.inferior = inferior
        self.superior = superior

    def fit(self, X, y=None):
        X = pd.DataFrame(X).astype(float)
        # Los límites se guardan como arreglos de numpy (no como Series de pandas) para que el modelo
        # guardado no dependa de la versión de pandas ni de pyarrow al abrirlo en otra máquina
        self.limites_ = (X.quantile(self.inferior).to_numpy(), X.quantile(self.superior).to_numpy())
        return self

    def transform(self, X):
        X = pd.DataFrame(X).to_numpy(dtype=float)
        # np.clip conserva los faltantes (NaN), que luego imputa el paso siguiente
        return np.clip(X, self.limites_[0], self.limites_[1])
