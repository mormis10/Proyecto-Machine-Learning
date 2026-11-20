import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss

from features import ROOT, MODEL_COLS
from models import get_xy, make_gboost
from splits import walk_forward_splits

FIG = ROOT / "docs/figures"
LABELS = ["A", "D", "H"]
COLORES = {"baseline": "tab:gray", "logreg": "tab:blue", "gboost": "tab:orange"}


def plot_logloss_por_temporada(res):
    """Log loss por temporada de cada modelo + diferencia GBM - LogReg."""
    tabla = res.pivot(index="season", columns="modelo", values="log_loss")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    for m in ["baseline", "logreg", "gboost"]:
        a1.plot(tabla.index, tabla[m], marker="o", label=m, color=COLORES[m])
    a1.set(title="Log loss por temporada de prueba (menor es mejor)", xlabel="Temporada")
    a1.legend()
    diff = tabla["gboost"] - tabla["logreg"]
    a2.bar(diff.index, diff, color=np.where(diff < 0, "tab:orange", "tab:blue"))
    a2.axhline(0, color="k", lw=0.8)
    a2.set(title="GBM − LogReg (negativo = gana GBM)", xlabel="Temporada")
    fig.tight_layout()
    fig.savefig(FIG / "04_logloss_por_temporada.png", dpi=130)
    plt.close(fig)


def plot_calibracion(preds, n_bins=10):
    """Curva de calibración de cada clase (H, D, A) para cada modelo."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, c in zip(axes, ["H", "D", "A"]):
        for m in ["logreg", "gboost"]:
            sub = preds[preds["modelo"] == m]
            obs, pred = calibration_curve(sub["FTR"] == c, sub[f"p_{c}"],
                                          n_bins=n_bins, strategy="quantile")
            ax.plot(pred, obs, marker="o", label=m, color=COLORES[m])
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set(title=f"Calibración clase {c}", xlabel="Probabilidad predicha",
               ylabel="Frecuencia observada")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "05_calibracion.png", dpi=130)
    plt.close(fig)


def tabla_subgrupos_elo(preds):
    """Log loss por tercil de |elo_diff|: partidos parejos vs. dispares."""
    preds = preds.assign(grupo=pd.qcut(preds["elo_diff"].abs(), 3,
                                       labels=["parejo", "medio", "dispar"]))
    filas = []
    for (g, m), sub in preds.groupby(["grupo", "modelo"], observed=True):
        P = sub[[f"p_{c}" for c in LABELS]].to_numpy()
        filas.append(dict(grupo=g, modelo=m, n=len(sub),
                          log_loss=log_loss(sub["FTR"], P, labels=LABELS)))
    return pd.DataFrame(filas).pivot(index="grupo", columns="modelo", values="log_loss")


def plot_importancias(feats):
    """Feature importance de GBM entrenado en el último fold."""
    ultimo = list(walk_forward_splits(feats))[-1]
    X, y = get_xy(feats, ultimo.train_idx)
    clf = make_gboost().fit(X, y).named_steps["clf"]
    imp = pd.Series(clf.feature_importances_, index=MODEL_COLS).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.index, imp, color="tab:orange")
    ax.set(title=f"Importancia de features (GBM, train < {ultimo.test_season})")
    fig.tight_layout()
    fig.savefig(FIG / "06_importancia_gbm.png", dpi=130)
    plt.close(fig)
    return imp


def main():
    res = pd.read_csv(ROOT / "docs/results_by_season.csv")
    preds = pd.read_parquet(ROOT / "data/processed/predictions.parquet")
    feats = pd.read_parquet(ROOT / "data/processed/features.parquet")

    plot_logloss_por_temporada(res)
    plot_calibracion(preds)
    sub = tabla_subgrupos_elo(preds)
    sub.to_csv(ROOT / "docs/results_by_elo_group.csv")
    print("Log loss por subgrupo de |elo_diff|:\n", sub.round(4))
    imp = plot_importancias(feats)
    print("\nTop 5 features GBM:\n", imp.sort_values(ascending=False).head().round(3))


if __name__ == "__main__":
    main()
