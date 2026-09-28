"""NOS v1 IsolationForest. Fibonacci fuera del vector. Sin nos_combined publico."""
from __future__ import annotations
from pathlib import Path
import numpy as np

DESCRIPTOR_KEYS = ("fractal_d2", "lacunarity", "beta", "anisotropy", "entropy", "euler")
MODEL_PATH = Path(__file__).resolve().parent / "models" / "nos_models.joblib"
GLOBAL_HPARAMS = {"n_estimators": 200, "contamination": 0.02, "max_samples": 256, "max_features": 1.0, "random_state": 42, "bootstrap": False}
TILE_HPARAMS = {"n_estimators": 120, "contamination": 0.08, "max_samples": "auto", "random_state": 42}

def _finite(x, default=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default

def vectorize(plugin_results):
    def g(*keys, default=0.0):
        cur = plugin_results
        for k in keys:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(k, default)
        return _finite(cur, default)
    return np.array([
        g("fractal_base", "d2", default=1.6),
        g("fractal_base", "lacunarity", default=0.5),
        g("kolmogorov_1941", "beta", default=2.0),
        g("anisotropy", "anisotropy_index"),
        g("entropy", "normalized_entropy", default=0.2),
        g("persistent_homology", "euler_characteristic"),
    ], dtype=float)

def synthetic_sky_corpus(n=400, seed=42):
    rng = np.random.default_rng(seed)
    X = np.column_stack([
        rng.normal(1.85, 0.18, n), rng.normal(0.45, 0.18, n), rng.normal(2.20, 0.55, n),
        rng.beta(2.0, 6.0, n), rng.beta(4.0, 3.0, n), rng.normal(0.0, 8.0, n),
    ])
    X[:, 0] = np.clip(X[:, 0], 1.1, 2.2)
    X[:, 1] = np.clip(X[:, 1], 0.05, 1.5)
    X[:, 2] = np.clip(X[:, 2], 0.3, 4.5)
    return X

def fit_nos(corpus_vectors, contamination=None):
    from sklearn.covariance import EllipticEnvelope
    from sklearn.ensemble import IsolationForest
    X = np.asarray(corpus_vectors, dtype=float)
    if X.ndim != 2 or X.shape[0] < 10:
        raise ValueError("corpus demasiado pequeno")
    hp = dict(GLOBAL_HPARAMS)
    if contamination is not None:
        hp["contamination"] = contamination
    iso = IsolationForest(**hp).fit(X)
    rob = None
    try:
        rob = EllipticEnvelope(random_state=42, support_fraction=0.8).fit(X)
    except Exception:
        rob = None
    return iso, rob

def save_models(models, path=MODEL_PATH):
    import joblib
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "hparams": GLOBAL_HPARAMS}, path)
    return path

def load_models(path=MODEL_PATH):
    path = Path(path)
    if not path.exists():
        return None
    import joblib
    blob = joblib.load(path)
    if isinstance(blob, dict) and "models" in blob:
        return blob["models"]
    return blob

_CACHE = {"models": None, "source": None}

def get_models():
    if _CACHE["models"] is not None:
        return _CACHE["models"], _CACHE["source"]
    loaded = load_models()
    if loaded is not None:
        _CACHE["models"] = loaded; _CACHE["source"] = "corpus_joblib"
        return loaded, "corpus_joblib"
    models = fit_nos(synthetic_sky_corpus())
    _CACHE["models"] = models; _CACHE["source"] = "synthetic_sky"
    return models, "synthetic_sky"

def nos_score(models, v):
    iso, rob = models
    x = np.asarray(v, dtype=float).reshape(1, -1)
    s_iso = float(-iso.decision_function(x)[0])
    pred = int(iso.predict(x)[0])
    out = {
        "nos_isoforest": s_iso,
        "isoforest_outlier": pred == -1,
        "isoforest_predict": pred,
        "hparams": GLOBAL_HPARAMS,
        "vector_keys": list(DESCRIPTOR_KEYS),
    }
    if rob is not None:
        out["nos_mahalanobis"] = float(-rob.decision_function(x)[0])
    return out

def score_with_isoforest(plugin_results):
    models, source = get_models()
    emp = nos_score(models, vectorize(plugin_results))
    emp["available"] = True
    emp["background"] = source
    emp["version"] = "v1_isoforest"
    emp["vector"] = vectorize(plugin_results).tolist()
    emp["note"] = "iso y mahalanobis por separado. Fibonacci fuera del vector. Fondo=%s." % source
    return emp

def morphological_nos_v1(plugin_results, prior_fallback=None):
    base = dict(prior_fallback or {})
    try:
        emp = score_with_isoforest(plugin_results)
        base["empirical"] = emp
        base["nos_isoforest"] = emp["nos_isoforest"]
        base["isoforest_outlier"] = emp["isoforest_outlier"]
        base["note"] = emp["note"]
    except Exception as exc:
        base["empirical"] = {"available": False, "error": str(exc)}
    return base

def tile_isolation_forest(image, n=12):
    from sklearn.ensemble import IsolationForest
    from scipy import stats
    img = np.asarray(image, dtype=np.float64)
    if img.ndim > 2:
        img = img.mean(axis=2)
    h, w = img.shape
    th, tw = max(4, h // n), max(4, w // n)
    feats = []
    for i in range(n):
        for j in range(n):
            p = img[i*th:(i+1)*th, j*tw:(j+1)*tw].ravel()
            p = p[np.isfinite(p)]
            if p.size < 8:
                feats.append([0, 0, 3, 0])
            else:
                feats.append([float(p.mean()), float(p.std()), float(stats.kurtosis(p, fisher=False)) if p.std() > 0 else 3.0, float(np.mean(np.abs(np.diff(p))))])
    X = np.asarray(feats, dtype=float)
    iso = IsolationForest(**TILE_HPARAMS).fit(X)
    raw = -iso.decision_function(X)
    pred = iso.predict(X)
    heat = raw.reshape(n, n)
    heat = (heat - heat.min()) / (np.ptp(heat) + 1e-12)
    iy, ix = np.unravel_index(int(np.argmax(heat)), heat.shape)
    return {"heatmap": heat, "n_outliers": int((pred == -1).sum()), "peak_tile": [int(iy), int(ix)], "peak_score": float(heat[iy, ix]), "method": "isolation_forest_tiles", "grid": n, "hparams": TILE_HPARAMS}
