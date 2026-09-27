"""NOS morfologico v0 + gancho v1 empirico."""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

CANON = {
    "psf_star": np.array([1.15, 3.4, 0.25, 0.05, 0.25, 0.05]),
    "cirrus": np.array([1.55, 2.4, 0.20, 0.05, 0.55, 0.25]),
    "filament": np.array([1.45, 2.2, 0.70, 0.08, 0.45, 0.20]),
    "sky_noise": np.array([1.90, 0.4, 0.25, 0.10, 0.95, 0.80]),
    "detector_grid": np.array([1.50, 0.8, 0.55, 0.70, 0.40, 0.30]),
}

def _unit(v):
    def g(*keys, default=0.0):
        cur = v
        for k in keys:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(k, default)
        try:
            x = float(cur)
            return x if x == x else default
        except (TypeError, ValueError):
            return default
    return np.array([
        g("fractal_base", "d0", default=1.5),
        g("kolmogorov_1941", "beta", default=1.5),
        g("anisotropy", "anisotropy_index"),
        g("periodicity", "periodicity_score"),
        g("entropy", "normalized_entropy", default=0.5),
        min(g("persistent_homology", "betti_0") / 20.0, 1.0),
    ], dtype=float)

def _scale(vec):
    out = vec.copy()
    out[0] = np.clip((vec[0] - 0.8) / 1.4, 0, 1)
    out[1] = np.clip(vec[1] / 5.0, 0, 1)
    return out

def morphological_nos(plugin_results):
    x = _scale(_unit(plugin_results))
    dists = []
    for name, proto in CANON.items():
        dists.append((name, float(np.linalg.norm(x - _scale(proto)))))
    dists.sort(key=lambda t: t[1])
    nearest, dmin = dists[0]
    nos = float(np.clip(1.0 - dmin / 1.8, 0.0, 1.0))
    out = {
        "nos": nos,
        "nearest_class": nearest,
        "distance_to_nearest": dmin,
        "ranking": [{"class": n, "distance": d} for n, d in dists],
        "interpretation": "Se parece a una clase canonica." if nos >= 0.55 else "Lejos del prior v0.",
        "note": "v0 manual. v1 si existe models/nos_models.joblib",
    }
    try:
        from nos_empirical import morphological_nos_v1
        out = morphological_nos_v1(plugin_results, prior_fallback=out)
    except Exception:
        pass
    return out
