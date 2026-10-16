import pytest
from splits import walk_forward_splits, TEST_SEASONS


def test_folds_esperados(feats):
    folds = list(walk_forward_splits(feats))
    assert [f.test_season for f in folds] == TEST_SEASONS == list(range(2020, 2026))


def test_sin_solape_temporal(feats):
    for f in walk_forward_splits(feats):
        tr, te = feats.loc[f.train_idx], feats.loc[f.test_idx]
        assert tr["Date"].max() < te["Date"].min()
        assert set(f.train_idx).isdisjoint(f.test_idx)
        assert (te["SeasonStart"] == f.test_season).all()
        assert (tr["SeasonStart"] < f.test_season).all()


def test_validacion_interna(feats):
    for f in walk_forward_splits(feats):
        tr = feats.loc[f.train_idx]
        inner, val = feats.loc[f.inner_train_idx], feats.loc[f.val_idx]
        assert (val["SeasonStart"] == f.test_season - 1).all()
        assert (val["SeasonStart"] != f.test_season).all()
        assert inner["Date"].max() < val["Date"].min()
        assert set(f.inner_train_idx) | set(f.val_idx) == set(f.train_idx)
        assert set(f.inner_train_idx).isdisjoint(f.val_idx)


def test_2020_marcada_atipica(feats):
    por_temporada = {f.test_season: f for f in walk_forward_splits(feats)}
    assert por_temporada[2020].test_atypical
    assert por_temporada[2021].val_atypical        # 2021/22 valida en 2020/21
    assert not any(f.test_atypical for t, f in por_temporada.items() if t != 2020)
    assert not any(f.val_atypical for t, f in por_temporada.items() if t != 2021)


def test_assert_detecta_fuga(feats):
    malo = feats.copy()
    idx = malo.index[malo["SeasonStart"] == 2019][0]
    malo.loc[idx, "Date"] = malo["Date"].max() + __import__("pandas").Timedelta(days=30)
    with pytest.raises(AssertionError):
        list(walk_forward_splits(malo))