import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score

from splits import walk_forward_splits
from models import get_xy, make_logreg, make_gboost
from itertools import product


LABELS = ["A", "D", "H"]   

GRID_LOGREG = [dict(C=c) for c in [0.001, 0.01, 0.1, 1.0, 10.0]]
GRID_GBOOST = [dict(max_depth=d, n_estimators=n, learning_rate=0.05)
               for d, n in product([1, 2, 3], [100, 200, 400])]

def brier(y, P):
    """Brier multiclase: suma de errores al cuadrado contra el one-hot, promediado."""
    Y = pd.get_dummies(pd.Categorical(y, categories=LABELS)).to_numpy(float)
    return np.mean(np.sum((P - Y) ** 2, axis=1))


def metricas(y, P):
    pred = np.array(LABELS)[P.argmax(axis=1)]
    return dict(log_loss=log_loss(y, P, labels=LABELS),
                brier=brier(y, P),
                accuracy=accuracy_score(y, pred))

def elegir_params(make_fn, grid, feats, fold):
    """Entrena cada combinación con inner_train y elige la de menor log loss en val.
    La temporada de prueba no participa."""
    Xin, yin = get_xy(feats, fold.inner_train_idx)
    Xval, yval = get_xy(feats, fold.val_idx)
    scores = []
    for params in grid:
        m = make_fn(**params).fit(Xin, yin)
        scores.append(log_loss(yval, m.predict_proba(Xval), labels=LABELS))
    i = int(np.argmin(scores))
    return grid[i], scores[i]


def main():
    feats = pd.read_parquet("data/processed/features.parquet")
    filas, preds = [], []

    for f in walk_forward_splits(feats):
        Xtr, ytr = get_xy(feats, f.train_idx)
        Xte, yte = get_xy(feats, f.test_idx)

        
        freq = ytr.value_counts(normalize=True).reindex(LABELS).to_numpy()
        P = {"baseline": np.tile(freq, (len(yte), 1))}

        
        elegidos = {}
        for nombre, make_fn, grid in [("logreg", make_logreg, GRID_LOGREG),
                                      ("gboost", make_gboost, GRID_GBOOST)]:
            params, val_ll = elegir_params(make_fn, grid, feats, f)
            modelo = make_fn(**params).fit(Xtr, ytr)
            assert list(modelo.classes_) == LABELS
            P[nombre] = modelo.predict_proba(Xte)
            elegidos[nombre] = dict(params=str(params), val_log_loss=val_ll)
            print(f"{f.test_season} {nombre}: {params}")

        for nombre, p in P.items():
            
            filas.append(dict(season=f.test_season, modelo=nombre,
                        atipica=f.test_atypical, **metricas(yte, p),
                        **elegidos.get(nombre, {})))
            preds.append(pd.DataFrame(p, columns=[f"p_{c}" for c in LABELS], index=f.test_idx)
                         .assign(modelo=nombre, season=f.test_season,
                                 FTR=yte, elo_diff=Xte["elo_diff"]))

    res = pd.DataFrame(filas)
    res.to_csv("docs/results_by_season.csv", index=False)
    pd.concat(preds).to_parquet("data/processed/predictions.parquet")

    print(res.pivot(index="season", columns="modelo", values="log_loss").round(4))
    print(res.groupby("modelo")[["log_loss", "brier", "accuracy"]].mean().round(4))


if __name__ == "__main__":
    main()
