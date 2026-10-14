import pandas as pd
m = pd.read_parquet("data/interim/matches_concat.parquet")

equipos = pd.concat([m["HomeTeam"], m["AwayTeam"]])
print(equipos.nunique())
print(sorted(equipos.unique()))
print(equipos.value_counts().sort_values().head(15))   

print("Forza Milan")


largo = pd.concat([
    m[["Season", "HomeTeam"]].rename(columns={"HomeTeam": "Team"}),
    m[["Season", "AwayTeam"]].rename(columns={"AwayTeam": "Team"}),
])
conteo = largo.groupby(["Season", "Team"]).size()
print(conteo[~conteo.isin([38, 42])])   

