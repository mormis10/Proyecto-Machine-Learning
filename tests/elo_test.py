import pytest
from features import compute_elo


@pytest.fixture(scope="module")
def elo(m):
    return compute_elo(m).set_index("match_id")


def test_primera_jornada_en_1500(m, elo):
    primer_dia = m["Date"] == m["Date"].min()
    assert (elo.loc[primer_dia.values, ["elo_home", "elo_away"]] == 1500).all().all()


def test_ascendidos_empiezan_igual(m, elo):
    # los 3 ascendidos de 2024/25 deben arrancar con el mismo rating
    j = m.reset_index(drop=True).join(elo)
    s = j[j["SeasonStart"] == 2024]
    for equipo in ["Leicester", "Ipswich", "Southampton"]:
        primero = s[(s["HomeTeam"] == equipo) | (s["AwayTeam"] == equipo)].iloc[0]
        r = primero["elo_home"] if primero["HomeTeam"] == equipo else primero["elo_away"]
        assert r == pytest.approx(1368, abs=1)


def test_sin_fuga(m, elo):
    # cambiar el resultado de un partido NO debe alterar el Elo previo a ese partido
    k = 8000
    m2 = m.reset_index(drop=True).copy()
    m2.loc[k, "FTR"] = "A" if m2.loc[k, "FTR"] != "A" else "H"
    elo2 = compute_elo(m2).set_index("match_id")
    assert elo.loc[:k].equals(elo2.loc[:k])          # hasta k inclusive: idéntico
    assert not elo.loc[k + 1:].equals(elo2.loc[k + 1:])  # después sí cambia
