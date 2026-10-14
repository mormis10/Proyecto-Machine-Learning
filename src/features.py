from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data/interim/matches_clean.parquet"
TRAIN_START = 2000   # 93/94-99/00

SHOT_COLS = ["shots_for", "shots_against", "sot_for", "sot_against"]


def to_long(m):
    """Un partido -> dos filas, una por equipo, desde su perspectiva."""
    m = m.reset_index(drop=True).rename_axis("match_id").reset_index()
    base = m[["match_id", "Season", "SeasonStart", "Date"]]

    home = base.assign(
        team=m["HomeTeam"], opponent=m["AwayTeam"], is_home=1,
        gf=m["FTHG"], ga=m["FTAG"],
        shots_for=m["HS"], shots_against=m["AS"],
        sot_for=m["HST"], sot_against=m["AST"],
    )
    away = base.assign(
        team=m["AwayTeam"], opponent=m["HomeTeam"], is_home=0,
        gf=m["FTAG"], ga=m["FTHG"],
        shots_for=m["AS"], shots_against=m["HS"],
        sot_for=m["AST"], sot_against=m["HST"],
    )

    long = pd.concat([home, away], ignore_index=True)
    long["points"] = np.select([long["gf"] > long["ga"], long["gf"] == long["ga"]], [3, 1], 0)
    return (
        long.sort_values(["Date", "match_id", "is_home"], ascending=[True, True, False], kind="stable")
        .reset_index(drop=True)
    )

ELO_INIT, ELO_K, ELO_HOME, ELO_CARRY = 1500, 20, 60, 0.8
RESULT = {"H": 1.0, "D": 0.5, "A": 0.0}


def expected_home(r_home, r_away):
    return 1 / (1 + 10 ** ((r_away - (r_home + ELO_HOME)) / 400))


def compute_elo(m):
    """Elo de cada equipo ANTES de cada partido, en una fila por partido."""
    m = m.reset_index(drop=True)
    elo_home = np.empty(len(m))
    elo_away = np.empty(len(m))
    ratings, prev_teams = {}, set()

    for season in sorted(m["SeasonStart"].unique()):
        games = m[m["SeasonStart"] == season]
        teams = set(games["HomeTeam"]) | set(games["AwayTeam"])

        if not prev_teams:   # primera temporada: todos en 1500
            ratings = {t: ELO_INIT for t in teams}
        else:
            ratings = {t: ELO_INIT + ELO_CARRY * (r - ELO_INIT) for t, r in ratings.items()}
            promoted_rating = np.mean([ratings[t] for t in prev_teams - teams])
            for t in teams - prev_teams:
                ratings[t] = promoted_rating

        for i, home, away, ftr in zip(games.index, games["HomeTeam"], games["AwayTeam"], games["FTR"]):
            rh, ra = ratings[home], ratings[away]
            elo_home[i], elo_away[i] = rh, ra          # 1) guardar ANTES
            delta = ELO_K * (RESULT[ftr] - expected_home(rh, ra))
            ratings[home], ratings[away] = rh + delta, ra - delta   # 2) actualizar DESPUÉS

        prev_teams = teams

    return pd.DataFrame({
        "match_id": m.index,
        "elo_home": elo_home,
        "elo_away": elo_away,
        "elo_diff": elo_home - elo_away,
    })

FORM_WINDOW = 5
FORM_COLS = ["points", "gf", "ga"]


def rolling_prev(long, by, cols, window=FORM_WINDOW):
    """Promedio de los últimos `window` partidos ANTERIORES de cada grupo."""
    return long.groupby(by)[cols].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )


def add_form(long):
    long = long.copy()
    overall = rolling_prev(long, "team", FORM_COLS)
    venue = rolling_prev(long, ["team", "is_home"], FORM_COLS)
    long[[f"form_{c}" for c in FORM_COLS]] = overall.to_numpy()
    long[[f"form_venue_{c}" for c in FORM_COLS]] = venue.to_numpy()
    return long


def add_shots(long):
    long = long.copy()
    rolled = rolling_prev(long, "team", SHOT_COLS)
    long[[f"roll_{c}" for c in SHOT_COLS]] = rolled.to_numpy()
    return long


if __name__ == "__main__":
    long = to_long(pd.read_parquet(CLEAN))
    print(long.head(10))
    print(long.shape)
