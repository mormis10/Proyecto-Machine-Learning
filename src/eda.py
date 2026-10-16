import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import CLEAN, ROOT, MODEL_COLS, build_features

FIG = ROOT / "docs/figures"


def plot_resultados_por_temporada(m):
    """% de H/D/A por temporada y tendencia de la ventaja de local (%H - %A)."""
    pct = pd.crosstab(m["SeasonStart"], m["FTR"], normalize="index").mul(100)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    for c, color in zip(["H", "D", "A"], ["tab:blue", "tab:gray", "tab:red"]):
        a1.plot(pct.index, pct[c], label=c, color=color)
    a1.set(title="Resultados por temporada (%)", xlabel="Temporada (año de inicio)")
    a1.legend()
    a2.plot(pct.index, pct["H"] - pct["A"], color="tab:green")
    a2.axvline(2020, ls="--", color="k", lw=0.8)
    a2.set(title="Ventaja de local: %H - %A (línea: 2020/21, sin público)", xlabel="Temporada")
    fig.tight_layout()
    fig.savefig(FIG / "01_resultados_por_temporada.png", dpi=130)
    plt.close(fig)


def plot_victoria_vs_elo(feats, min_n=30):
    """P(ganar) según Elo propio - Elo rival, separada por jugar de local o de visitante."""
    local = pd.DataFrame({"d": feats["elo_diff"], "gana": (feats["FTR"] == "H").astype(int)})
    visita = pd.DataFrame({"d": -feats["elo_diff"], "gana": (feats["FTR"] == "A").astype(int)})
    bins = np.arange(-400, 401, 50)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for nombre, df, color in [("De local", local, "tab:blue"), ("De visitante", visita, "tab:red")]:
        g = df.groupby(pd.cut(df["d"], bins), observed=False)["gana"].agg(["mean", "count"])
        g = g[g["count"] >= min_n]
        centros = [iv.mid for iv in g.index]
        ax.plot(centros, g["mean"], marker="o", label=nombre, color=color)
    ax.axvline(0, color="k", lw=0.6)
    ax.set(title="P(ganar) según diferencia de Elo (propio - rival)",
           xlabel="Diferencia de Elo", ylabel="Proporción de victorias")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "02_victoria_vs_elo.png", dpi=130)
    plt.close(fig)


def plot_correlacion(feats):
    corr = feats[MODEL_COLS].corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlación entre features del modelo")
    fig.tight_layout()
    fig.savefig(FIG / "03_correlacion.png", dpi=130)
    plt.close(fig)
    return corr


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    m = pd.read_parquet(CLEAN)
    feats = build_features(m)
    plot_resultados_por_temporada(m)
    plot_victoria_vs_elo(feats)
    corr = plot_correlacion(feats)
    # pares muy correlacionados, para comentar en el informe
    pares = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack()
    print(pares[pares.abs() > 0.7].sort_values(key=abs, ascending=False).round(2).to_string())
    print(f"Figuras en {FIG.relative_to(ROOT)}")