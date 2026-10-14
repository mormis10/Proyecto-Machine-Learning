from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/interim/matches_concat.parquet"


@pytest.fixture(scope="module")
def m():
    return pd.read_parquet(DATA)


def test_partidos_por_temporada(m):
    conteo = m.groupby("Season").size()
    esperado = {"1993/94": 462, "1994/95": 462}
    raros = {s: n for s, n in conteo.items() if n != esperado.get(s, 380)}
    assert not raros, f"Temporadas con conteo inesperado: {raros}"


def test_equipos_por_temporada(m):
    n = m.groupby("Season")["HomeTeam"].nunique()
    esperado = n.index.map(lambda s: 22 if s in ("1993/94", "1994/95") else 20)
    assert (n.values == esperado).all(), f"Número de equipos incorrecto:\n{n[n.values != esperado]}"


def test_ftr_coherente_con_goles(m):
    esperado = np.select([m["FTHG"] > m["FTAG"], m["FTHG"] < m["FTAG"]], ["H", "A"], "D")
    malos = m[m["FTR"] != esperado]
    assert malos.empty, f"FTR no coincide con los goles:\n{malos[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]}"


def test_sin_duplicados(m):
    dup = m[m.duplicated(["Season", "HomeTeam", "AwayTeam"], keep=False)]
    assert dup.empty, f"Partidos duplicados:\n{dup[['Season', 'Date', 'HomeTeam', 'AwayTeam']]}"


def test_local_distinto_de_visitante(m):
    assert (m["HomeTeam"] != m["AwayTeam"]).all()


def test_sin_nulos_en_columnas_clave(m):
    clave = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    nulos = m[clave].isna().sum()
    assert nulos.sum() == 0, f"Nulos encontrados:\n{nulos[nulos > 0]}"
