from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def field_powerlaw(n=128, beta=2.0, seed=0):
    rng = np.random.default_rng(seed)
    ky = np.fft.fftfreq(n)[:, None]
    kx = np.fft.fftfreq(n)[None, :]
    k = np.sqrt(kx * kx + ky * ky)
    k[0, 0] = 1.0
    amp = k ** (-beta / 2.0)
    amp[0, 0] = 0
    phase = rng.uniform(0, 2 * np.pi, (n, n))
    img = np.fft.ifft2(amp * np.exp(1j * phase)).real
    return (img - img.min()) / (np.ptp(img) + 1e-12)

def field_lattice(n=128, period=8):
    x = np.arange(n)
    g = np.sin(2 * np.pi * x / period)
    return 0.5 + 0.5 * np.outer(g, g)

def test_beta_recovers():
    from plugins.kolmogorov_1941 import Kolmogorov1941
    r = Kolmogorov1941().analyze(field_powerlaw(beta=2.5, seed=1))
    assert abs(r["beta"] - 2.5) < 1.2, r
    assert "k62_kurtosis" in r

def test_lattice_periodicity():
    from plugins.periodicity import analyze_periodicity
    r = analyze_periodicity(field_lattice())
    assert r.get("periodicity_score", 0) >= 0.15 or r.get("n_significant_peaks", 0) >= 2

def test_white_noise_not_lattice():
    from plugins.periodicity import analyze_periodicity
    r = analyze_periodicity(np.random.default_rng(3).random((96, 96)))
    assert r.get("periodicity_score", 1) < 0.85

def test_fdr_and_bh():
    from scoring import benjamini_hochberg, fdr_decision
    mask, pc = benjamini_hochberg([0.001, 0.4, 0.8])
    assert mask[0]
    assert fdr_decision({"a": 0.001, "b": 0.5})["fdr_pass"] is True

def test_schema_validate():
    from schema.validate_candidate import validate
    assert validate({"schema": "nope"})
    good = {"schema": "cosmic-candidate.v0.1", "source": {"filename": "x.fits", "is_fits": True}, "morphology": {"is_candidate": False}, "spectrum": {"status": "missing"}}
    assert validate(good) == []

if __name__ == "__main__":
    test_beta_recovers(); test_lattice_periodicity(); test_white_noise_not_lattice(); test_fdr_and_bh(); test_schema_validate()
    print("ALL_SYNTHETIC_OK")
