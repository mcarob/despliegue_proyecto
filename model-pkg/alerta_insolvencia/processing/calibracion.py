"""Calibración del puntaje y niveles de alerta.

El random forest se entrena con class_weight balanceado: ordena bien a las empresas, pero sus "probabilidades"
quedan infladas (en promedio 0,08 frente a una tasa real de 0,003). La calibración de Platt convierte el
puntaje en una probabilidad en la escala real: p = 1 / (1 + exp(-(a * logit(puntaje) + b))).
Los coeficientes a y b se estiman con predicciones fuera de tiempo (años de validación), nunca con la prueba.
"""

import typing as t

import numpy as np
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def ajustar_platt(puntaje: np.ndarray, y: np.ndarray) -> t.Dict[str, float]:
    """Estima a y b con una regresión logística de una sola variable (logit del puntaje), sin regularizar."""
    reg = LogisticRegression(C=1e6, max_iter=1000).fit(logit(puntaje).reshape(-1, 1), y)
    return {"coeficiente": float(reg.coef_[0, 0]), "intercepto": float(reg.intercept_[0])}


def calibrar(puntaje: np.ndarray, coeficiente: float, intercepto: float) -> np.ndarray:
    """Probabilidad calibrada de abrir insolvencia el año siguiente."""
    return 1 / (1 + np.exp(-(coeficiente * logit(puntaje) + intercepto)))


def nivel_alerta(puntaje: np.ndarray, umbral_alto: float, umbral_medio: float) -> t.List[str]:
    """Alto = dentro del 10 % de mayor puntaje del año de referencia; medio = el 20 % siguiente; bajo = el resto."""
    puntaje = np.asarray(puntaje, dtype=float)
    return np.where(puntaje >= umbral_alto, "alto", np.where(puntaje >= umbral_medio, "medio", "bajo")).tolist()
