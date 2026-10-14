import numpy as np
import pandas as pd
import pytest
from features import to_long, add_form, add_shots, to_wide, add_diffs, TEAM_COLS


@pytest.fixture(scope="module")
def long(m):
    return add_shots(add_form(to_long(m)))


@pytest.fixture(scope="module")
def wide(long):
    return add_diffs(to_wide(long, TEAM_COLS))


def test_una_fila_por_partido(m, wide):
    assert len(wide) == len(m)
    assert wide.index.equals(pd.RangeIndex(len(m)))


def test_columnas_esperadas(wide):
    for c in TEAM_COLS:
        assert {f"home_{c}", f"away_{c}", f"diff_{c}"} <= set(wide.columns)


def test_diferencia_coincide_con_formato_largo(long, wide):
    # para un partido cualquiera, la diff debe salir de las dos filas del formato largo
    k = 9000
    filas = long[long["match_id"] == k].set_index("is_home")
    esperado = filas.loc[1, "form_points"] - filas.loc[0, "form_points"]
    assert np.isclose(wide.loc[k, "diff_form_points"], esperado)
