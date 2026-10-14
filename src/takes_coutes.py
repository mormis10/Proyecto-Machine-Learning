from pathlib import Path
import pandas as pd

ESPERADAS = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR",
             "Referee", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"}

for p in sorted(Path("data/raw").glob("season-*.csv")):
    cols = set(pd.read_csv(p, nrows=0, encoding="latin-1").columns)
    extra = cols - ESPERADAS
    if extra:
        print(p.name, sorted(extra))
