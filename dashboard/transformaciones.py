"""Transformación propia del modelo (igual a la de experiments/entrenar_modelos.py).

Va en un archivo aparte para que el modelo guardado con joblib se pueda abrir desde cualquier script.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RecorteExtremos(BaseEstimator, TransformerMixin):
    """Recorta cada razón a sus percentiles 1 y 99 calculados en entrenamiento (las razones tienen colas enormes)."""

    def __init__(self, inferior=0.01, superior=0.99):
        self.inferior = inferior
        self.superior = superior

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.limites_ = (X.quantile(self.inferior), X.quantile(self.superior))
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.clip(self.limites_[0], self.limites_[1], axis=1).to_numpy()
