import numpy as np
import pytest
from features import (to_long, add_form, add_shots, add_flag, fill_prior,
                      FORM_WINDOW)


def _pipeline(m, rellenar):
    long = add_flag(add_shots(add_form(to_long(m))))
    return fill_prior(long) if rellenar else long


@pytest.fixture(scope="module")
def crudo(m):
    return _pipeline(m, rellenar=False)


@pytest.fixture(scope="module")
def relleno(m):
    return _pipeline(m, rellenar=True)


def test_primer_partido_marcado(relleno):
    primeros = relleno.groupby("team").head(1)
    assert (primeros["pocos_partidos"] == 1).all()


def test_bandera_recalculo(relleno):
    rng = np.random.default_rng(1)
    for i in rng.choice(len(relleno), 200, replace=False):
        r = relleno.iloc[i]
        prev = relleno[(relleno["team"] == r["team"]) & (relleno["Date"] < r["Date"])]
        esperado = int(len(prev) < FORM_WINDOW or prev["shots_for"].notna().sum() < FORM_WINDOW)
        assert esperado == r["pocos_partidos"], f"fila {i}"


def test_prior_recalculo(crudo, relleno):
    """Donde la forma era NaN, el valor es la media de la liga con fechas ANTERIORES."""
    primera_fecha = crudo["Date"].min()
    nan_pts = crudo[crudo["form_points"].isna() & (crudo["Date"] > primera_fecha)]
    nan_venue = crudo[crudo["form_venue_gf"].isna() & (crudo["Date"] > primera_fecha)]
    assert len(nan_pts) > 0 and len(nan_venue) > 0

    for idx in nan_pts.index[:: max(1, len(nan_pts) // 30)]:
        r = crudo.loc[idx]
        esperado = crudo.loc[crudo["Date"] < r["Date"], "points"].mean()
        assert np.isclose(relleno.loc[idx, "form_points"], esperado)

    for idx in nan_venue.index[:: max(1, len(nan_venue) // 30)]:
        r = crudo.loc[idx]
        previos = crudo[(crudo["Date"] < r["Date"]) & (crudo["is_home"] == r["is_home"])]
        assert np.isclose(relleno.loc[idx, "form_venue_gf"], previos["gf"].mean())


def test_prior_no_toca_valores_reales(crudo, relleno):
    """Solo se rellenan NaN: lo que ya tenía valor queda idéntico."""
    for c in ["form_points", "form_venue_gf", "roll_shots_for"]:
        ok = crudo[c].notna()
        assert crudo.loc[ok, c].equals(relleno.loc[ok, c])


def test_prior_sin_fuga(m, relleno):
    """Perturbación: alterar un partido no cambia bandera ni valores de partidos anteriores."""
    k = 8000
    m2 = m.reset_index(drop=True).copy()
    m2.loc[k, ["FTHG", "FTAG"]] = m2.loc[k, ["FTAG", "FTHG"]].to_numpy() + [5, 0]
    relleno2 = _pipeline(m2, rellenar=True)
    cols = [c for c in relleno.columns if c.startswith(("form_", "roll_"))] + ["pocos_partidos"]
    hasta_k = relleno["match_id"] <= k
    assert relleno.loc[hasta_k, cols].equals(relleno2.loc[hasta_k, cols])