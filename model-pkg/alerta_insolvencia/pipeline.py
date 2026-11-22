"""Pipeline del modelo: preprocesamiento y random forest en un solo objeto de scikit-learn.

Es la misma configuración de la corrida random_forest_hoja50 de la Entrega 2 (registrada en MLflow) y
del modelo v1.0 del tablero, así los resultados son comparables entre entregas.
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from alerta_insolvencia.config.core import config
from alerta_insolvencia.processing.transformaciones import RecorteExtremos

MC = config.modelo


def construir_pipeline() -> Pipeline:
    """Recorte de extremos y mediana para las razones, moda para los indicadores, one-hot para el subsector."""
    preprocesar = ColumnTransformer([
        ("razones", Pipeline([
            ("recorte", RecorteExtremos(MC.recorte_inferior, MC.recorte_superior)),
            ("imputar", SimpleImputer(strategy="median")),
        ]), MC.razones),
        ("indicadores", SimpleImputer(strategy="most_frequent"), MC.indicadores),
        # Un subsector que no se vio en entrenamiento queda en ceros en vez de romper la predicción
        ("ciiu", OneHotEncoder(handle_unknown="ignore"), MC.categoricas),
    ])
    bosque = RandomForestClassifier(
        n_estimators=MC.n_estimators,
        min_samples_leaf=MC.min_samples_leaf,
        class_weight=MC.class_weight,  # compensa que solo 0,3 % a 0,8 % de las empresas sean eventos
        n_jobs=-1,
        random_state=MC.random_state,
    )
    return Pipeline([("preprocesar", preprocesar), ("modelo", bosque)])
