import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from evaluate import LABELS, brier, metricas, elegir_params
from models import get_xy, make_logreg
from significance import logloss_por_partido
from splits import walk_forward_splits


def _fold(feats, test_season=2022):
    return next(f for f in walk_forward_splits(feats) if f.test_season == test_season)


def test_brier_casos_conocidos():
    y = pd.Series(["H", "D", "A"])
    perfecta = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)  
    assert brier(y, perfecta) == 0
    uniforme = np.full((3, 3), 1 / 3)
    assert np.isclose(brier(y, uniforme), 2 / 3)


def test_baseline_accuracy_es_siempre_local():
    y = pd.Series(["H", "H", "D", "A", "H"])
    P = np.tile([0.3, 0.25, 0.45], (len(y), 1))   
    assert metricas(y, P)["accuracy"] == (y == "H").mean()


def test_probabilidades_validas(feats):
    f = _fold(feats)
    m = make_logreg().fit(*get_xy(feats, f.train_idx))
    assert list(m.classes_) == LABELS
    P = m.predict_proba(get_xy(feats, f.test_idx)[0])
    assert np.allclose(P.sum(axis=1), 1)
    assert ((P > 0) & (P < 1)).all()


def test_logloss_por_partido_coincide_con_sklearn():
    rng = np.random.default_rng(0)
    P = rng.dirichlet([1, 1, 1], size=50)
    y = rng.choice(LABELS, size=50)
    preds = pd.DataFrame(P, columns=["p_A", "p_D", "p_H"]).assign(FTR=y)
    assert np.isclose(logloss_por_partido(preds)["ll"].mean(), log_loss(y, P, labels=LABELS))


def test_ajuste_no_ve_la_temporada_de_prueba(feats):
    f = _fold(feats)
    grid = [dict(C=c) for c in [0.01, 1.0]]
    antes = elegir_params(make_logreg, grid, feats, f)

    alterado = feats.copy()
    alterado.loc[f.test_idx, "FTR"] = "A"
    despues = elegir_params(make_logreg, grid, alterado, f)
    assert antes == despues
