from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW, INTERIM = ROOT / "data/raw", ROOT / "data/interim"

def season_label(code):  # "9394" -> (1993, "1993/94"), "0001" -> (2000, "2000/01")
    start = int(code[:2]) + (1900 if int(code[:2]) >= 93 else 2000)
    return start, f"{start}/{code[2:]}"

frames = []
for p in sorted(RAW.glob("season-*.csv")):
    df = pd.read_csv(p, encoding="latin-1")
    df = df.dropna(how="all").loc[:, ~df.columns.str.startswith("Unnamed")]
    start, label = season_label(p.stem.split("-")[1])
    df.insert(0, "Season", label)
    df.insert(1, "SeasonStart", start)
    frames.append(df)

matches = pd.concat(frames, ignore_index=True).sort_values("SeasonStart", kind="stable")
INTERIM.mkdir(parents=True, exist_ok=True)
matches.to_parquet(INTERIM / "matches_concat.parquet", index=False)
print(matches.groupby("Season").size())