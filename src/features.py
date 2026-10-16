from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data/interim/matches_clean.parquet"
OUT = ROOT / "data/processed/features.parquet"
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


TEAM_COLS = [f"form_{c}" for c in FORM_COLS] + [f"form_venue_{c}" for c in FORM_COLS] \
          + [f"roll_{c}" for c in SHOT_COLS]


FLAG_COLS = ["pocos_partidos"]
RAW_COLS = FORM_COLS + SHOT_COLS


def add_flag(long):
    """pocos_partidos = 1 si a ese equipo todavía no le cabe una ventana completa (5 partidos)
    de historia, ya sea de resultados o de tiros. Solo mira partidos ANTERIORES."""
    long = long.copy()
    n_prev = long.groupby("team").cumcount()                       # partidos previos del equipo
    tiene_tiros = long["shots_for"].notna().astype(int)
    n_tiros_prev = tiene_tiros.groupby(long["team"]).cumsum() - tiene_tiros   # previos CON estadística de tiros
    long["pocos_partidos"] = ((n_prev < FORM_WINDOW) | (n_tiros_prev < FORM_WINDOW)).astype(int)
    return long


def _prior_por_fecha(df, cols):
    """Media de la liga de cada columna usando SOLO partidos de fechas estrictamente anteriores.
    Devuelve un DataFrame indexado por Date."""
    suma = df.groupby("Date")[cols].sum()
    n = df.groupby("Date")[cols].count()
    return suma.cumsum().shift(1) / n.cumsum().shift(1)


def fill_prior(long):
    """Rellena los NaN de form_*, form_venue_* y roll_* con la media de la liga hasta la fecha
    anterior al partido. Debe llamarse DESPUÉS de add_form/add_shots (nunca antes del rolling)."""
    long = long.copy()
    fechas = long["Date"]

    def en_filas(prior, c):
        return pd.Series(prior[c].reindex(fechas).to_numpy(), index=long.index)

    prior_all = _prior_por_fecha(long, RAW_COLS)
    prior_home = _prior_por_fecha(long[long["is_home"] == 1], FORM_COLS)
    prior_away = _prior_por_fecha(long[long["is_home"] == 0], FORM_COLS)

    for c in FORM_COLS:
        long[f"form_{c}"] = long[f"form_{c}"].fillna(en_filas(prior_all, c))
        por_sede = pd.Series(
            np.where(long["is_home"] == 1, en_filas(prior_home, c), en_filas(prior_away, c)),
            index=long.index,
        )
        long[f"form_venue_{c}"] = long[f"form_venue_{c}"].fillna(por_sede)
    for c in SHOT_COLS:
        long[f"roll_{c}"] = long[f"roll_{c}"].fillna(en_filas(prior_all, c))
    return long


def to_wide(long, cols):
    """Dos filas por partido -> una, con prefijos home_ / away_."""
    home = long[long["is_home"] == 1].set_index("match_id")[cols].add_prefix("home_")
    away = long[long["is_home"] == 0].set_index("match_id")[cols].add_prefix("away_")
    return home.join(away)


def add_diffs(wide, cols=TEAM_COLS):
    wide = wide.copy()
    for c in cols:
        wide[f"diff_{c}"] = wide[f"home_{c}"] - wide[f"away_{c}"]
    return wide


MODEL_COLS = (
    ["elo_diff"]
    + [f"diff_{c}" for c in TEAM_COLS]
    + [f"{lado}_{f}" for f in FLAG_COLS for lado in ("home", "away")]
)

# Días desde el primer partido con los que se tolera quedarse sin valor previo
# (la primera jornada con estadísticas de tiros no tiene ningún partido anterior).
MAX_DIAS_SIN_PRIOR = 14


def _quitar_sin_prior(feats):
    """Descarta partidos que siguen con NaN tras el valor previo. Solo se tolera en la
    primera jornada con datos; si ocurre en otro lado es un bug y se detiene con error."""
    sin = feats[MODEL_COLS].isna().any(axis=1)
    if sin.any():
        limite = feats["Date"].min() + pd.Timedelta(days=MAX_DIAS_SIN_PRIOR)
        fuera = feats.loc[sin & (feats["Date"] > limite)]
        assert fuera.empty, (
            f"{len(fuera)} partidos sin valor previo fuera de la primera jornada; "
            f"columnas con NaN: {fuera[MODEL_COLS].isna().any().loc[lambda s: s].index.tolist()}"
        )
    return feats.loc[~sin]


def build_features(m, impute=True, burn_in=True):
    """matches_clean -> una fila por partido con target, Elo y diferencias de forma/tiros.

    impute=False  : deja los NaN (para el reporte de calidad).
    burn_in=False : conserva 93/94-99/00 (para el reporte de calidad).
    El filtro de burn-in va AL FINAL: Elo y forma se calculan con todas las temporadas.
    """
    m = m.reset_index(drop=True)                      # el índice es el match_id
    long = add_flag(add_shots(add_form(to_long(m))))
    if impute:
        long = fill_prior(long)

    wide = add_diffs(to_wide(long, TEAM_COLS + FLAG_COLS))
    elo = compute_elo(m).set_index("match_id")
    base = m[["Season", "SeasonStart", "Date", "FTR"]].rename_axis("match_id")
    feats = base.join(elo).join(wide)

    if burn_in:
        feats = feats[feats["SeasonStart"] >= TRAIN_START]
    if impute and burn_in:
        feats = _quitar_sin_prior(feats)
    return feats


if __name__ == "__main__":
    feats = build_features(pd.read_parquet(CLEAN))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(OUT)
    print(f"{OUT.relative_to(ROOT)}: {feats.shape[0]} partidos, {feats.shape[1]} columnas")