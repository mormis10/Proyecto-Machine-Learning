from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import MODEL_COLS


def get_xy(feats, idx):
    """X (siempre MODEL_COLS, en el mismo orden) e y (FTR) de los partidos `idx`."""
    sub = feats.loc[idx]
    return sub[MODEL_COLS], sub["FTR"]


def make_logreg(C=1.0):
    """Regresión logística: ESCALA, porque es sensible a la escala de las variables.
    El scaler vive dentro del Pipeline, así que .fit(X_train) lo ajusta solo con train."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=1000)),
    ])


def make_gboost(**params):
    """Gradient Boosting: los árboles no necesitan escalado, por eso no lleva StandardScaler."""
    base = dict(n_estimators=200, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=0)
    base.update(params)
    return Pipeline([("clf", GradientBoostingClassifier(**base))])