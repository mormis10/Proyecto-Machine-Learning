import numpy as np
import pytest
from features import to_long, add_form


@pytest.fixture(scope="module")
def form(m):
    return add_form(to_long(m))


def test_primer_partido_sin_forma(form):
    primeros = form.groupby("team").head(1)
    assert primeros["form_points"].isna().all()


def test_recalculo_manual(form):
    rng = np.random.default_rng(0)
    for i in rng.choice(len(form), 200, replace=False):
        r = form.iloc[i]
        prev = form[(form["team"] == r["team"]) & (form["Date"] < r["Date"])].tail(5)
        esperado = prev["points"].mean() if len(prev) else np.nan
        assert np.isclose(esperado, r["form_points"], equal_nan=True), f"fila {i}"

        prev_v = prev_v = form[(form["team"] == r["team"]) & (form["is_home"] == r["is_home"])
                               & (form["Date"] < r["Date"])].tail(5)
        esperado_v = prev_v["gf"].mean() if len(prev_v) else np.nan
        assert np.isclose(esperado_v, r["form_venue_gf"], equal_nan=True), f"fila {i}"


def test_sin_fuga(m, form):
    k = 8000
    m2 = m.reset_index(drop=True).copy()
    m2.loc[k, ["FTHG", "FTAG"]] = m2.loc[k, ["FTAG", "FTHG"]].to_numpy() + [5, 0]
    form2 = add_form(to_long(m2))
    cols = [c for c in form.columns if c.startswith("form_")]
    hasta_k = form["match_id"] <= k
    assert form.loc[hasta_k, cols].equals(form2.loc[hasta_k, cols])
