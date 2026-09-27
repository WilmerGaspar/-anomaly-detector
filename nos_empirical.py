"""NOS v1 empirico: IsolationForest + EllipticEnvelope."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

DESCRIPTOR_KEYS = ("fractal_d2", "lacunarity", "beta", "anisotropy", "entropy", "euler", "phi_score")
MODEL_PATH = Path(__file__).resolve().parent / "models" / "nos_models.joblib"

def vectorize(plugin_results):
    def g(*keys, default=0.0):
        cur = plugin_results
        for k in keys:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(k, default)
        try:
            x = float(cur)
            return x if np.isfinite(x) else default
        except (TypeError, ValueError):
            return default
    return [g("fractal_base", "d2", default=1.6), g("fractal_base", "lacunarity", default=0.5), g("kolmogorov_1941", "beta", default=2.0), g("anisotropy", "anisotropy_index"), g("entropy", "normalized_entropy", default=0.2), g("persistent_homology", "euler_characteristic"), g("fibonacci", "fibonacci_score")]

def fit_nos(corpus_vectors):
    from sklearn.covariance import EllipticEnvelope
    from sklearn.ensemble import IsolationForest
    X = np.asarray(corpus_vectors, dtype=float)
    if X.ndim != 2 or X.shape[0] < 10:
        raise ValueError("corpus demasiado pequeno")
    iso = IsolationForest(contamination=0.05, random_state=42).fit(X)
    rob = EllipticEnvelope(random_state=42, support_fraction=0.8).fit(X)
    return iso, rob

def save_models(models, path=MODEL_PATH):
    import joblib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, path)
    return path

def load_models(path=MODEL_PATH):
    path = Path(path)
    if not path.exists():
        return None
    import joblib
    return joblib.load(path)

def nos_score(models, v):
    iso, rob = models
    x = np.asarray(v, dtype=float).reshape(1, -1)
    s_iso = float(-iso.decision_function(x)[0])
    s_rob = float(-rob.decision_function(x)[0])
    return {"nos_isoforest": s_iso, "nos_mahalanobis": s_rob, "nos_combined": float((s_iso + s_rob) / 2.0), "version": "v1_empirical"}

def morphological_nos_v1(plugin_results, prior_fallback=None):
    models = load_models()
    base = dict(prior_fallback or {})
    if models is None:
        base["empirical"] = {"available": False, "note": "python scripts/build_corpus.py && python scripts/fit_nos.py corpus"}
        return base
    try:
        emp = nos_score(models, vectorize(plugin_results))
        emp["available"] = True
        base["empirical"] = emp
        base["nos_v1"] = emp["nos_combined"]
        base["note"] = "NOS v1 IsolationForest + EllipticEnvelope"
    except Exception as exc:
        base["empirical"] = {"available": False, "error": str(exc)}
    return base
