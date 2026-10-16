from dataclasses import dataclass
import numpy as np

TEST_SEASONS = list(range(2020, 2026))   # 2020/21 ... 2025/26
ATYPICAL_SEASONS = {2020}                # 2020/21: sin público


@dataclass
class Fold:
    test_season: int
    train_idx: np.ndarray        # match_id de todo lo anterior a la temporada de prueba
    test_idx: np.ndarray
    inner_train_idx: np.ndarray  # train sin su última temporada (para ajustar hiperparámetros)
    val_idx: np.ndarray          # última temporada del bloque de entrenamiento
    val_season: int
    test_atypical: bool
    val_atypical: bool


def walk_forward_splits(feats, test_seasons=TEST_SEASONS):
    """Genera un Fold por temporada de prueba. `feats` debe tener SeasonStart y Date."""
    for t in test_seasons:
        train = feats[feats["SeasonStart"] < t]
        test = feats[feats["SeasonStart"] == t]
        if train.empty or test.empty:
            raise ValueError(f"Temporada de prueba {t}: sin datos de train o de test")

        val_season = int(train["SeasonStart"].max())
        inner = train[train["SeasonStart"] < val_season]
        val = train[train["SeasonStart"] == val_season]

        # Garantías contra fuga temporal, en CADA fold
        assert train["Date"].max() < test["Date"].min(), f"fold {t}: train se mezcla con test"
        assert inner["Date"].max() < val["Date"].min(), f"fold {t}: inner_train se mezcla con val"

        yield Fold(
            test_season=t,
            train_idx=train.index.to_numpy(), test_idx=test.index.to_numpy(),
            inner_train_idx=inner.index.to_numpy(), val_idx=val.index.to_numpy(),
            val_season=val_season,
            test_atypical=t in ATYPICAL_SEASONS, val_atypical=val_season in ATYPICAL_SEASONS,
        )