import argparse
import ast
import sys

import pandas as pd

from features import CLEAN, OUT, ROOT, MODEL_COLS, build_features
from models import make_logreg, make_gboost
from team_names import normalize_team

RESULTS = ROOT / "docs/results_by_season.csv"


def partido_ficticio(m, home, away):
    """Agrega el partido a predecir al final, un día después del último partido.
    El marcador de relleno (0-0) no afecta sus features: todo se calcula con partidos ANTERIORES."""
    ultimo = m.iloc[-1]
    fila = dict(Season=ultimo["Season"], SeasonStart=ultimo["SeasonStart"],
                Date=ultimo["Date"] + pd.Timedelta(days=1),
                HomeTeam=home, AwayTeam=away, FTHG=0, FTAG=0, FTR="D")
    return pd.concat([m, pd.DataFrame([fila])], ignore_index=True)


def params_ultimo_fold():
    """Hiperparámetros elegidos por evaluate.py en la temporada de prueba más reciente."""
    res = pd.read_csv(RESULTS)
    ultima = res[res["season"] == res["season"].max()].set_index("modelo")
    return {m: ast.literal_eval(ultima.loc[m, "params"]) for m in ["logreg", "gboost"]}


def main():
    ap = argparse.ArgumentParser(description="Probabilidades H/D/A de un partido de la Premier League")
    ap.add_argument("home", help="equipo local, p. ej. 'Arsenal'")
    ap.add_argument("away", help="equipo visitante, p. ej. 'Man City'")
    args = ap.parse_args()

    m = pd.read_parquet(CLEAN)
    home, away = normalize_team(args.home), normalize_team(args.away)
    validos = set(m.loc[m["SeasonStart"] == m["SeasonStart"].max(), "HomeTeam"])
    for t in (home, away):
        if t not in validos:
            sys.exit(f"'{t}' no está en la última temporada. Equipos válidos:\n"
                     + ", ".join(sorted(validos)))
    if home == away:
        sys.exit("Local y visitante tienen que ser distintos.")

    
    x = build_features(partido_ficticio(m, home, away)).iloc[[-1]][MODEL_COLS]

    
    feats = pd.read_parquet(OUT)
    params = params_ultimo_fold()
    filas = {}
    for nombre, make_fn in [("logreg", make_logreg), ("gboost", make_gboost)]:
        modelo = make_fn(**params[nombre]).fit(feats[MODEL_COLS], feats["FTR"])
        p = dict(zip(modelo.classes_, modelo.predict_proba(x)[0]))
        filas[nombre] = {"Local (H)": p["H"], "Empate (D)": p["D"], "Visita (A)": p["A"]}

    print(f"\n{home} (local) vs {away} (visita)  |  datos hasta {m['Date'].max():%d/%m/%Y}")
    print(f"elo_diff = {x['elo_diff'].iloc[0]:+.0f}\n")
    print(pd.DataFrame(filas).T.map(lambda v: f"{v:.1%}").to_string())


if __name__ == "__main__":
    main()
