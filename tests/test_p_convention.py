from scoring import fdr_decision, benjamini_hochberg

def test_fdr_scope_and_floor_note():
    p = {"aniso": 0.0769, "flatness": 0.0769, "entropy": 0.0769, "energy_mean": 0.0769}
    d = fdr_decision(p, alpha=0.05)
    assert d["scope"] == ["aniso", "flatness", "entropy", "energy_mean"]
    assert d["fdr_pass"] is False
    assert d["n_passed"] == 0

def test_bh_rejects_when_small():
    mask, crit = benjamini_hochberg([0.001, 0.02, 0.8], alpha=0.05)
    assert mask[0]
