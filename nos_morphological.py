"""NOS morfologico: rareza del vector de descriptores respecto a un prior canonico."""
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


def _unit(v: Dict) -> np.ndarray:
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

    d0 = g("fractal_base", "d0", default=1.5)
    beta = g("kolmogorov_1941", "beta", default=1.5)
    aniso = g("anisotropy", "anisotropy_index")
    per = g("periodicity", "periodicity_score")
    ent = g("entropy", "normalized_entropy", default=0.5)
    b0 = min(g("persistent_homology", "betti_0") / 20.0, 1.0)
    return np.array([d0, beta, aniso, per, ent, b0], dtype=float)


def _scale(vec: np.ndarray) -> np.ndarray:
    out = vec.copy()
    out[0] = np.clip((vec[0] - 0.8) / 1.4, 0, 1)
    out[1] = np.clip(vec[1] / 5.0, 0, 1)
    return out


def morphological_nos(plugin_results: Dict) -> Dict:
    x = _scale(_unit(plugin_results))
    dists: List[Tuple[str, float]] = []
    for name, proto in CANON.items():
        p = _scale(proto)
        d = float(np.linalg.norm(x - p))
        dists.append((name, d))
    dists.sort(key=lambda t: t[1])
    nearest, dmin = dists[0]
    nos = float(np.clip(1.0 - dmin / 1.8, 0.0, 1.0))
    return {
        "nos": nos,
        "nearest_class": nearest,
        "distance_to_nearest": dmin,
        "ranking": [{"class": n, "distance": d} for n, d in dists],
        "interpretation": (
            "Se parece a una clase canonica de cielo/instrumento."
            if nos >= 0.55
            else "Morfologia lejos del prior v0. Candidato *o* prior insuficiente."
        ),
        "note": "Prior v0 hecho a mano. Sustituir por Mahalanobis sobre 50 FITS reales.",
    }
