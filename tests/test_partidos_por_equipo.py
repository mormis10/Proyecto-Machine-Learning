from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/interim/matches_concat.parquet"

def test_partidos_por_equipo():
    m = pd.read_parquet(DATA)
    largo = pd.concat([
        m[["Season", "HomeTeam"]].rename(columns={"HomeTeam": "Team"}),
        m[["Season", "AwayTeam"]].rename(columns={"AwayTeam": "Team"}),
    ])
    conteo = largo.groupby(["Season", "Team"]).size()
    raros = conteo[~conteo.isin([38, 42])]
    assert raros.empty, f"Equipos con conteo anormal:\n{raros}"

