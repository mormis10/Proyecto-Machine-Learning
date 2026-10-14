import numpy as np
import pytest
from features import to_long, add_shots, TRAIN_START


@pytest.fixture(scope="module")
def shots(m):
    return add_shots(to_long(m))


def test_sin_tiros_antes_de_2000(shots):
    antes = shots[shots["SeasonStart"] < TRAIN_START]
    assert antes["roll_shots_for"].isna().all()


def test_recalculo_manual(shots):
    desde = shots[shots["SeasonStart"] >= TRAIN_START]
    rng = np.random.default_rng(1)
    for i in rng.choice(desde.index, 200, replace=False):
        r = shots.loc[i]
        prev = shots[(shots["team"] == r["team"]) & (shots["Date"] < r["Date"])].tail(5)
        esperado = prev["sot_for"].mean()   # mean() ignora NaN, igual que rolling
        assert np.isclose(esperado, r["roll_sot_for"], equal_nan=True), f"fila {i}"


def test_sin_fuga(m, shots):
    k = 8000
    m2 = m.reset_index(drop=True).copy()
    m2.loc[k, ["HS", "HST"]] = [99, 99]
    shots2 = add_shots(to_long(m2))
    cols = [c for c in shots.columns if c.startswith("roll_")]
    hasta_k = shots["match_id"] <= k
    assert shots.loc[hasta_k, cols].equals(shots2.loc[hasta_k, cols])
