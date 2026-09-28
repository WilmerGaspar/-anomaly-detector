"""Oraculos de los dos JSON reales. Si se rompe el state machine, falla aqui."""
from materials_map import interpret, decide_state

NIRCAM = {
    "fractal_base": {"d0": 1.96, "d2": 1.766, "lacunarity": 0.884, "multifractality_index": 0.636},
    "kolmogorov_1941": {"beta": 2.278, "intermittency_factor": 1.0, "isotropic_score": 0.0},
    "periodicity": {"periodicity_score": 0.174, "n_significant_peaks": 1, "likely_instrument_artifact": False},
    "anisotropy": {"anisotropy_index": 0.270},
    "persistent_homology": {"betti_0": 6, "betti_1": 4, "euler_characteristic": 2},
    "renormalization_group": {"criticality_score": 0.5, "scale_invariance_score": 0.766},
    "entropy": {"normalized_entropy": 0.018},
}

ACS = {
    "fractal_base": {"d0": 1.881, "d2": 1.693, "lacunarity": 0.630, "multifractality_index": 0.673},
    "kolmogorov_1941": {"beta": 2.139, "intermittency_factor": 1.0, "isotropic_score": 0.0},
    "periodicity": {"periodicity_score": 0.150, "n_significant_peaks": 5, "likely_instrument_artifact": True, "axis_aligned_fraction": 0.7},
    "anisotropy": {"anisotropy_index": 0.238},
    "persistent_homology": {"betti_0": 3, "betti_1": 6, "euler_characteristic": -3},
    "renormalization_group": {"criticality_score": 0.5, "scale_invariance_score": 0.712},
    "entropy": {"normalized_entropy": 0.018},
}

USABLE = {"verdict": "usable_science", "is_fits": True, "product_level": "image_calibrated"}

def test_nircam_known_or_weak():
    m = interpret(NIRCAM, structure_z=90.0, p_value=0.04, fdr_pass=False, provenance=USABLE)
    assert m["state"] == "known_or_weak"
    assert m["is_candidate"] is False
    assert m["family_tie"] is True

def test_acs_known_or_weak_mixed():
    m = interpret(ACS, structure_z=60.0, p_value=0.04, fdr_pass=False, provenance=USABLE, metadata={"filter": "CLEAR1L"})
    assert m["state"] == "known_or_weak"
    assert m["is_candidate"] is False
    assert m["family_tie"] is True
    assert m["instrument_warning"] is True
    assert m["instrument_warning_reason"]

def test_fdr_pass_becomes_needs_spectrum():
    m = interpret(NIRCAM, structure_z=5.0, p_value=0.01, fdr_pass=True, provenance=USABLE)
    assert m["state"] in ("needs_spectrum", "morph_interesting")

def test_detector_is_exploratory():
    assert decide_state({"verdict": "usable_science", "product_level": "detector"}, fdr_pass=True) == "exploratory_only"

def test_reject_non_fits():
    assert decide_state({"verdict": "reject_for_discovery", "is_fits": False}, fdr_pass=True) == "reject"
