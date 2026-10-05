"""El estado y el siguiente paso siguen al semaforo (analysis_job.run_analysis).

NIRCam F187N real: semaforo 🔴 por artefacto del detector, pero el estado decia
"needs_spectrum" y proponia "Hollin, agregados de polvo" / "Extincion + hielos"."""
import json

import numpy as np

import analysis_job as aj


def test_detector_artifact_is_rejected_without_material_hint():
    rng = np.random.default_rng(0)
    n = 256
    xx = np.mgrid[:n, :n][1]
    raw = (rng.normal(size=(n, n)) + 3 * np.sin(2 * np.pi * xx / 8.0)).astype(np.float32)   # bandas del detector
    lo, hi = np.percentile(raw, [2, 98])
    crop = np.clip((raw - lo) / (hi - lo), 0, 1).astype(np.float32)
    out = aj.run_analysis(crop=crop, crop_raw=raw, valid=np.ones((n, n), bool),
                          active=["periodicity", "ridges", "anisotropy", "fractal_base"], n_null=19, seed=80,
                          name="jw0_nrcb1_i2d.fits",
                          meta_full={"is_fits": True, "format": "FITS", "instrument": "NIRCAM", "filter": "F187N",
                                     "empty_area_fraction": 0.0},
                          source={"archive": "MAST", "download_url": "https://mast.stsci.edu/x/jw0_nrcb1_i2d.fits"},
                          hole_frac=0.0, max_empty_fraction=0.001)
    card = json.loads(out["json"])
    assert card["discovery_gate"]["level"] == "invalid"
    assert card["state"] == card["morphology"]["state"] == "instrument_artifact"
    assert card["followup"]["action"] == aj.NEXT_STEP["instrument_artifact"]
    assert "instrument_artifact" in card["followup"]["reject_if"]
    assert card["morphology"]["is_candidate"] is False
