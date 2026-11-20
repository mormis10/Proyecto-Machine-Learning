import numpy as np
import pandas as pd

from features import ROOT

N_BOOT = 10_000


def logloss_por_partido(preds):
    """-log(probabilidad que el modelo le dio al resultado real), un valor por fila."""
    real = preds["FTR"].map({"A": 0, "D": 1, "H": 2}).to_numpy()
    P = preds[["p_A", "p_D", "p_H"]].to_numpy()
    p_real = P[np.arange(len(P)), real]
    return preds.assign(ll=-np.log(p_real))


def bootstrap_diff(d, n_boot=N_BOOT, seed=0):
    """Bootstrap de la media de d = ll_gboost - ll_logreg (pareado por partido)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    medias = d[idx].mean(axis=1)
    lo, hi = np.percentile(medias, [2.5, 97.5])
    return dict(n=len(d), diff=d.mean(), ic_bajo=lo, ic_alto=hi,
                frac_gbm_mejor=(medias < 0).mean())


def main():
    preds = logloss_por_partido(pd.read_parquet(ROOT / "data/processed/predictions.parquet"))

    ll = preds.pivot_table(index=preds.index, columns="modelo", values="ll")
    ll["season"] = preds.groupby(level=0)["season"].first()
    ll["d"] = ll["gboost"] - ll["logreg"]

    filas = [dict(season=s, **bootstrap_diff(g["d"].to_numpy()))
             for s, g in ll.groupby("season")]
    filas.append(dict(season="total", **bootstrap_diff(ll["d"].to_numpy())))

    res = pd.DataFrame(filas)
    res.to_csv(ROOT / "docs/significance_gbm_vs_logreg.csv", index=False)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
