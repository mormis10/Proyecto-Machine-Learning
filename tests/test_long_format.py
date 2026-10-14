import pytest
from features import to_long


@pytest.fixture(scope="module")
def long(m):
    return to_long(m)


def test_dos_filas_por_partido(m, long):
    assert len(long) == 2 * len(m)
    por_partido = long.groupby("match_id")["is_home"].agg(["size", "sum"])
    assert (por_partido["size"] == 2).all()   
    assert (por_partido["sum"] == 1).all()    


def test_perspectivas_espejadas(long):
    local = long[long["is_home"] == 1].set_index("match_id")
    visita = long[long["is_home"] == 0].set_index("match_id")
    assert (local["gf"] == visita["ga"]).all()
    assert (local["team"] == visita["opponent"]).all()
    assert local["shots_for"].equals(visita["shots_against"])


def test_puntos_por_partido(long):
    total = long.groupby("match_id")["points"].sum()
    assert total.isin([2, 3]).all()   
