import pandas as pd

from features import CLEAN, ROOT, MODEL_COLS, build_features

BINS = [1992, 1999, 2009, 2019, 2100]
LABELS = ["93/94-99/00 (burn-in)", "00/01-09/10", "10/11-19/20", "20/21-25/26"]
REPORT = ROOT / "docs/quality_report.txt"


def missing_by_era(df, cols):
    """% de NaN de cada columna en cada era."""
    era = pd.cut(df["SeasonStart"], bins=BINS, labels=LABELS)
    return df[cols].isna().groupby(era, observed=False).mean().mul(100).round(2).T


def class_by_season(df):
    """Conteo y % de H/D/A por temporada."""
    conteo = pd.crosstab(df["Season"], df["FTR"])[["H", "D", "A"]]
    pct = conteo.div(conteo.sum(axis=1), axis=0).mul(100).round(1).add_suffix("_%")
    return conteo.join(pct).assign(partidos=conteo.sum(axis=1))


def build_report(m):
    crudo = build_features(m, impute=False, burn_in=False)   # con NaN y con burn-in
    final = build_features(m)                                # lo que entra a los modelos
    fuente = ["HS", "AS", "HST", "AST"]
    partes = [
        ("1) Faltantes en columnas FUENTE de tiros (% por era)", missing_by_era(m, fuente)),
        ("2) Faltantes en features ANTES de imputar (% por era)", missing_by_era(crudo, MODEL_COLS)),
        ("3) Faltantes en features FINALES (% por era)", missing_by_era(final, MODEL_COLS)),
        (f"4) Partidos: {(m['SeasonStart'] >= 2000).sum()} desde 2000/01 -> "
         f"{len(final)} en features (descartados: {(m['SeasonStart'] >= 2000).sum() - len(final)})", None),
        ("5) Distribución de clases por temporada (features finales)", class_by_season(final)),
    ]
    out = []
    for titulo, tabla in partes:
        out.append(f"\n== {titulo} ==")
        if tabla is not None:
            out.append(tabla.to_string())
    return "\n".join(out)


if __name__ == "__main__":
    texto = build_report(pd.read_parquet(CLEAN))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(texto, encoding="utf-8")
    print(texto)