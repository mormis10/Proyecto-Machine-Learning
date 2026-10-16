import numpy as np
from features import MODEL_COLS, FLAG_COLS
from models import get_xy, make_logreg, make_gboost
from splits import walk_forward_splits


def _fold(feats, test_season=2022):
    return next(f for f in walk_forward_splits(feats) if f.test_season == test_season)


def test_mismas_columnas(feats):
    f = _fold(feats)
    Xtr, _ = get_xy(feats, f.train_idx)
    Xte, _ = get_xy(feats, f.test_idx)
    assert list(Xtr.columns) == list(Xte.columns) == MODEL_COLS


def test_sin_columnas_redundantes(feats):
    """diff_x = home_x - away_x: meter las tres sería colinealidad perfecta."""
    permitidas = {f"{l}_{f}" for f in FLAG_COLS for l in ("home", "away")}
    extra = [c for c in MODEL_COLS if c.startswith(("home_", "away_")) and c not in permitidas]
    assert extra == []


def test_sin_colinealidad_perfecta(feats):
    X = feats[MODEL_COLS].to_numpy(float)
    assert np.linalg.matrix_rank(X) == len(MODEL_COLS)


def test_scaler_se_ajusta_solo_con_train(feats):
    f = _fold(feats)
    Xtr, ytr = get_xy(feats, f.train_idx)
    Xall, _ = get_xy(feats, feats.index)
    pipe = make_logreg().fit(Xtr, ytr)
    scaler = pipe.named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, Xtr.mean().to_numpy())
    assert not np.allclose(scaler.mean_, Xall.mean().to_numpy())


def test_ambos_modelos_predicen(feats):
    f = _fold(feats)
    Xtr, ytr = get_xy(feats, f.train_idx)
    Xte, _ = get_xy(feats, f.test_idx)
    for model in (make_logreg(), make_gboost(n_estimators=30)):
        p = model.fit(Xtr, ytr).predict_proba(Xte)
        assert p.shape == (len(Xte), 3)
        np.testing.assert_allclose(p.sum(axis=1), 1.0)
        assert list(model.classes_) == ["A", "D", "H"]