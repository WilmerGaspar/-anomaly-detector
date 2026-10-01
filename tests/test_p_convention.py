from scoring import fdr_decision, benjamini_hochberg, FDR_SCOPE

def test_fdr_scope_and_floor_note():
    p = {k: 0.0769 for k in FDR_SCOPE}
    d = fdr_decision(p, alpha=0.05)
    assert d["scope"] == list(FDR_SCOPE)
    assert d["fdr_pass"] is False
    assert d["n_passed"] == 0

def test_bh_rejects_when_small():
    mask, crit = benjamini_hochberg([0.001, 0.02, 0.8], alpha=0.05)
    assert mask[0]
