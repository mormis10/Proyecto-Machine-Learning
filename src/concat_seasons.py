from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW, INTERIM = ROOT / "data/raw", ROOT / "data/interim"

from team_names import normalize_team

def season_label(code):  # "9394" -> (1993, "1993/94"), "0001" -> (2000, "2000/01")
    start = int(code[:2]) + (1900 if int(code[:2]) >= 93 else 2000)
    return start, f"{start}/{code[2:]}"

DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"]  # DataHub usa ISO; football-data.co.uk usa dd/mm/yy o dd/mm/yyyy

def parse_dates(s):
    out = pd.Series(pd.NaT, index=s.index)
    for fmt in DATE_FORMATS:
        out = out.fillna(pd.to_datetime(s, format=fmt, errors="coerce"))
    bad = s[out.isna()]
    if not bad.empty:
        raise ValueError(f"Fechas con formato desconocido: {bad.unique()[:5]}")
    return out


frames = []
for p in sorted(RAW.glob("season-*.csv")):
    df = pd.read_csv(p, encoding="latin-1")
    df = df.dropna(how="all").loc[:, ~df.columns.str.startswith("Unnamed")]
    df["Date"] = parse_dates(df["Date"].astype(str).str.strip())
    for col in ["HomeTeam", "AwayTeam"]:
        df[col] = df[col].map(normalize_team)
    start, label = season_label(p.stem.split("-")[1])
    df.insert(0, "Season", label)
    df.insert(1, "SeasonStart", start)
    frames.append(df)

matches = pd.concat(frames, ignore_index=True).sort_values("SeasonStart", kind="stable")
INTERIM.mkdir(parents=True, exist_ok=True)
matches.to_parquet(INTERIM / "matches_concat.parquet", index=False)
print(matches.groupby("Season").size())