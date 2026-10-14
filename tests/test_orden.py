from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/interim/matches_concat.parquet"

def test_orden_cronologico():
    m = pd.read_parquet(DATA)
    assert m["Date"].is_monotonic_increasing, "Hay fechas fuera de orden"
    assert m.index.equals(pd.RangeIndex(len(m))), "El índice no está reseteado"