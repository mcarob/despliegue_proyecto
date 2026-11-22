"""Métricas para una clase muy rara (0,2 % a 0,8 % de empresas abren insolvencia cada año).

La exactitud no sirve: un modelo que dice "nadie quiebra" acierta el 99,7 %. Se usan métricas de ordenamiento
(AUC, AUC-PR) y la lectura de negocio: qué parte de los eventos encuentra el banco si revisa el 10 % de las
empresas con mayor puntaje.
"""

from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def captura(y: np.ndarray, puntaje: np.ndarray, fraccion: float = 0.10) -> float:
    """Fracción de los eventos que quedan dentro del `fraccion` de empresas con mayor puntaje."""
    corte = np.quantile(puntaje, 1 - fraccion)
    return float(y[puntaje >= corte].sum() / y.sum())


def captura_por_anio(y: np.ndarray, puntaje: np.ndarray, anios: np.ndarray, fraccion: float = 0.10) -> float:
    """Igual que `captura`, pero el corte se calcula dentro de cada año, como trabajaría el banco:
    cada año revisa el 10 % de mayor riesgo de ese año."""
    capturados = 0
    for anio in np.unique(anios):
        m = anios == anio
        corte = np.quantile(puntaje[m], 1 - fraccion)
        capturados += y[m][puntaje[m] >= corte].sum()
    return float(capturados / y.sum())


def precision_en(y: np.ndarray, puntaje: np.ndarray, fraccion: float = 0.10) -> float:
    """Precisión de la lista de revisión: de las empresas revisadas, qué parte abre insolvencia."""
    corte = np.quantile(puntaje, 1 - fraccion)
    return float(y[puntaje >= corte].mean())


def curva_captura(y: np.ndarray, puntaje: np.ndarray, pasos: int = 50):
    """Puntos (fracción revisada, fracción de eventos capturados) para graficar la curva de captura."""
    orden = np.argsort(-puntaje)
    acumulado = np.cumsum(y[orden]) / y.sum()
    x = np.linspace(0, 1, pasos + 1)
    idx = np.clip((x * len(y)).astype(int) - 1, 0, len(y) - 1)
    return [0.0] + [float(v) for v in x[1:]], [0.0] + [float(v) for v in acumulado[idx[1:]]]


def metricas(
    y: np.ndarray,
    puntaje: np.ndarray,
    anios: Optional[np.ndarray] = None,
    probabilidad: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Conjunto completo de métricas que se registra en MLflow y se publica con el modelo."""
    y = np.asarray(y)
    puntaje = np.asarray(puntaje)
    prevalencia = float(y.mean())
    auc_pr = float(average_precision_score(y, puntaje))
    resultado = {
        "auc": float(roc_auc_score(y, puntaje)),
        "auc_pr": auc_pr,
        "auc_pr_sobre_prevalencia": auc_pr / prevalencia,
        "captura_top5": captura(y, puntaje, 0.05),
        "captura_top10": captura(y, puntaje, 0.10),
        "captura_top20": captura(y, puntaje, 0.20),
        "precision_top10": precision_en(y, puntaje, 0.10),
        "prevalencia": prevalencia,
    }
    if anios is not None:
        resultado["captura_top10_por_anio"] = captura_por_anio(y, puntaje, np.asarray(anios), 0.10)
    if probabilidad is not None:
        # El Brier solo tiene sentido con probabilidades calibradas (que estén en la escala de la tasa real)
        resultado["brier"] = float(brier_score_loss(y, probabilidad))
    return resultado


def intervalo_bootstrap(
    y: np.ndarray,
    puntaje: np.ndarray,
    funcion,
    repeticiones: int = 1000,
    semilla: int = 42,
    niveles: Sequence[float] = (0.025, 0.975),
):
    """Intervalo de confianza por remuestreo de una métrica. Con pocos eventos en la prueba (decenas),
    sirve para saber si la diferencia entre dos modelos es real o puede ser azar."""
    rng = np.random.default_rng(semilla)
    n = len(y)
    valores = []
    for _ in range(repeticiones):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0:
            continue
        valores.append(funcion(y[idx], puntaje[idx]))
    return [float(v) for v in np.quantile(valores, niveles)]
