from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data/interim/matches_clean.parquet"
EN_CURSO = {"2026/27"}   

def test_partidos_por_equipo():
    m = pd.read_parquet(CLEAN)
    largo = pd.concat([
        m[["Season", "HomeTeam"]].rename(columns={"HomeTeam": "Team"}),
        m[["Season", "AwayTeam"]].rename(columns={"AwayTeam": "Team"}),
    ])
    conteo = largo.groupby(["Season", "Team"]).size()
    en_curso = conteo.index.get_level_values("Season").isin(EN_CURSO)
    raros = conteo[(~en_curso & ~conteo.isin([38, 42])) | (en_curso & (conteo > 38))]
    assert raros.empty, f"Equipos con conteo anormal:\n{raros}"

