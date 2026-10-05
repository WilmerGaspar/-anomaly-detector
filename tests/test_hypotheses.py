"""Tabla de hipótesis (hypotheses.build): solo con 🟢/🟣, reglas físicas y pregunta abierta."""
import pytest

from hypotheses import COMPATIBLE, NOT_COMPATIBLE, NOT_MEASURABLE, build


def _card(beta=2.3, r2=0.97, flat=(36.0, 35.0), q95=(35.0, 30.0), ridge=-0.004, passed=("aniso",), aniso=0.22,
          b1=0, lac=7.4, mf=0.03, periodic=False, target="NGC 1333"):
    return {"source": {"target": target},
            "analytics": {"spectrum": {"fit": {"beta": beta, "r2": r2}},
                          "scales": {"flatness": list(flat), "flatness_null_q95": list(q95)}},
            "structure_test": {"fdr": {"passed_descriptors": list(passed)}},
            "descriptors": {"ridges": {"filament_excess": ridge}, "anisotropy": {"anisotropy_index": aniso},
                            "persistent_homology": {"betti_1": b1},
                            "fractal_base": {"lacunarity": lac, "multifractality_index": mf},
                            "periodicity": {"n_significant_peaks": 1 if periodic else 0, "lattice_consistent": periodic,
                                            "likely_instrument_artifact": False}}}


def _status(H):
    return {r["key"]: r["status"] for r in H["rows"]}


@pytest.mark.parametrize("level", ["none", "explained", "unconfirmed", "invalid", None])
def test_only_with_green_or_purple(level):
    H = build(_card(), level)
    assert H["active"] is False and "🟢" in H["note"]


def test_real_nircam_green_case():
    # Medidas del NIRCam F200W real que dio 🟢 (beta 2.32, intermitencia y anisotropía significativas).
    s = _status(build(_card(), "robust"))
    assert s["supersonic"] == s["magnetic"] == COMPATIBLE
    assert s["k41"] == s["filaments"] == s["shells"] == NOT_COMPATIBLE
    assert s["collapse"] == s["projection"] == s["unresolved"] == s["galaxies"] == NOT_MEASURABLE


def test_kolmogorov_filaments_and_shells():
    s = _status(build(_card(beta=3.6, ridge=0.02, b1=2, passed=()), "pioneer"))
    assert s["k41"] == s["filaments"] == s["shells"] == COMPATIBLE and s["magnetic"] == NOT_COMPATIBLE


def test_open_question_when_nothing_fits():
    H = build(_card(beta=1.2, flat=(1, 1), passed=(), lac=1.1), "robust")
    assert H["classification"] == "open_question" and "otro filtro" in H["summary"]
    assert not any(r["status"] == COMPATIBLE for r in H["rows"])


def test_bad_fit_is_not_measurable_and_context_always_present():
    H = build(_card(r2=0.5, target=""), "robust")
    s = _status(H)
    assert s["k41"] == s["supersonic"] == NOT_MEASURABLE
    assert "no indica el objeto" in H["context"] and "galaxias" in H["context"]
    assert all(r["question"] and r["refs"] for r in H["rows"])
