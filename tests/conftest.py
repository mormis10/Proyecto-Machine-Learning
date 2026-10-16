from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data/interim/matches_clean.parquet"


@pytest.fixture(scope="session")
def m():
    return pd.read_parquet(CLEAN)


@pytest.fixture(scope="session")
def feats(m):
    """features.parquet en memoria (sin escribir archivos)."""
    from features import build_features
    return build_features(m)