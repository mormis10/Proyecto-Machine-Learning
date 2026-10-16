import numpy as np
from features import build_features, MODEL_COLS, TRAIN_START


def test_burn_in_aplicado(feats):
    assert feats["SeasonStart"].min() == TRAIN_START


def test_sin_nan_en_modelo(feats):
    assert not feats[MODEL_COLS].isna().any().any()


def test_indice_match_id(feats):
    assert feats.index.name == "match_id"
    assert feats.index.is_unique


def test_target_y_metadatos(feats):
    assert {"FTR", "Season", "SeasonStart", "Date"} <= set(feats.columns)
    assert set(feats["FTR"].unique()) <= {"H", "D", "A"}


def test_pocos_descartados(m, feats):
    """Solo se pierde la primera jornada con estadísticas (no hay partido previo para el prior)."""
    esperados = (m["SeasonStart"] >= TRAIN_START).sum()
    assert 0 <= esperados - len(feats) <= 20


def test_diff_es_local_menos_visitante(feats):
    assert np.allclose(feats["diff_form_points"],
                       feats["home_form_points"] - feats["away_form_points"])


def test_elo_se_calcula_con_burn_in(m, feats):
    """Si el Elo se calculara filtrando antes, arrancaría en 1500 en 2000/01."""
    primera = feats[feats["SeasonStart"] == TRAIN_START].iloc[0]
    assert not np.isclose(primera["elo_home"], 1500) or not np.isclose(primera["elo_away"], 1500)


def test_sin_imputar_conserva_nan(m):
    crudo = build_features(m, impute=False, burn_in=False)
    assert crudo["home_form_points"].isna().any()
    assert len(crudo) == len(m)