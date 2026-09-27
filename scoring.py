"""Score global, nulo de fase, FDR y nulos adaptativos."""
from __future__ import annotations
import numpy as np

FEATURE_GETTERS = {
    "fractal_base": lambda r: r.get("complexity_score", np.nan),
    "kolmogorov_1941": lambda r: 0.6 * r.get("intermittency_factor", 0.0) + 0.4 * r.get("turbulence_intensity", 0.0),
    "lyapunov_stability": lambda r: r.get("chaos_strength", np.nan),
    "persistent_homology": lambda r: r.get("complexity_score", np.nan),
    "renormalization_group": lambda r: r.get("criticality_score", np.nan),
    "anisotropy": lambda r: r.get("anisotropy_index", np.nan),
    "periodicity": lambda r: r.get("periodicity_score", np.nan),
    "entropy": lambda r: float(np.clip(float(r.get("mutual_information_approx", 0.0)) / 4.0, 0.0, 1.0)),
}

def extract_feature_vector(plugin_results):
    names, vals = [], []
    for key, getter in FEATURE_GETTERS.items():
        if key not in plugin_results:
            continue
        try:
            v = float(getter(plugin_results[key]))
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            names.append(key); vals.append(v)
    return names, np.array(vals, dtype=float)

def compute_global_score(plugin_results, mode="balanced"):
    _, vals = extract_feature_vector(plugin_results)
    score = 0.5 if len(vals) == 0 else float(np.clip(np.mean(np.clip(vals, 0.0, 1.0)), 0.0, 1.0))
    if mode == "explorer": score = min(score * 1.15, 1.0)
    elif mode == "conservative": score = score * 0.85
    return float(np.clip(score, 0.0, 1.0))

def phase_randomized_surrogate(image, rng):
    img = np.asarray(image, dtype=np.float64)
    f = np.fft.rfft2(img)
    mag = np.abs(f)
    rand = rng.uniform(0.0, 2.0 * np.pi, size=f.shape)
    rand[0, 0] = 0.0
    surr = np.fft.irfft2(mag * np.exp(1j * rand), s=img.shape).real
    if surr.std() > 0 and img.std() > 0:
        surr = (surr - surr.mean()) / surr.std() * img.std() + img.mean()
    return np.clip(surr, float(img.min()), float(img.max()))

def downsample_for_null(image, max_side=96):
    h, w = image.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return image
    try:
        from skimage.transform import resize
        scale = max_side / side
        return resize(image, (max(16, int(h * scale)), max(16, int(w * scale))), anti_aliasing=True, preserve_range=True).astype(np.float64)
    except Exception:
        step = max(1, side // max_side)
        return image[::step, ::step]

def surrogate_null_test(image, observed_score, analyze_fn, n_simulations=40, seed=None):
    rng = np.random.default_rng(seed)
    small = downsample_for_null(image)
    null_scores = []
    for _ in range(n_simulations):
        try:
            null_scores.append(float(analyze_fn(phase_randomized_surrogate(small, rng))))
        except Exception:
            continue
    if not null_scores:
        return {"p_value": 1.0, "stars": "ns", "is_significant": False, "n_simulations": 0, "method": "phase-randomized surrogates", "null_scores": [], "z_score": 0.0}
    null = np.array(null_scores, dtype=float)
    p_value = max(float(np.mean(null >= observed_score)), 1.0 / (len(null) + 1))
    null_mean = float(null.mean())
    null_std = float(null.std(ddof=1)) if len(null) > 1 else 0.0
    z = (observed_score - null_mean) / (null_std + 1e-12)
    stars = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    return {"p_value": p_value, "stars": stars, "is_significant": p_value < 0.05, "null_mean": null_mean, "null_std": null_std, "n_simulations": int(len(null)), "method": "phase-randomized surrogates (espectro conservado)", "z_score": float(z), "null_scores": null.tolist()}

def cheap_score_from_image(image):
    img = np.asarray(image, dtype=np.float64)
    if img.ndim > 2:
        img = img.mean(axis=2)
    g = np.gradient(img)
    energy = g[0] ** 2 + g[1] ** 2
    aniso = float(np.clip((energy.std() / (energy.mean() + 1e-12)) / 3.0, 0.0, 1.0))
    bs = max(4, min(img.shape) // 8)
    masses = [float(img[i:i+bs, j:j+bs].sum()) for i in range(0, img.shape[0], bs) for j in range(0, img.shape[1], bs)]
    m = np.array(masses)
    lac = float(np.clip((m.var() / ((m.mean() ** 2) + 1e-12)) / 4.0, 0.0, 1.0)) if m.size else 0.0
    inc = np.diff(img, axis=1).ravel(); inc = inc - inc.mean()
    m2 = np.mean(inc ** 2); m4 = np.mean(inc ** 4)
    flat = m4 / (m2 ** 2) if m2 > 1e-18 else 3.0
    inter = float(np.clip((flat - 3.0) / 6.0, 0.0, 1.0))
    return float(np.clip(0.4 * aniso + 0.3 * lac + 0.3 * inter, 0.0, 1.0))

def benjamini_hochberg(p_values, alpha=0.05):
    p = np.asarray(p_values, dtype=float)
    p = p[np.isfinite(p)]
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool), 1.0
    order = np.argsort(p)
    ranked = p[order]
    threshold = (np.arange(1, m + 1) / m) * alpha
    below = ranked <= threshold
    if not below.any():
        return np.zeros(m, dtype=bool), 1.0
    k_max = int(np.max(np.where(below)[0]))
    rejected = np.zeros(m, dtype=bool)
    rejected[order[: k_max + 1]] = True
    return rejected, float(ranked[k_max])

def _cheap_metrics(image):
    img = np.asarray(image, dtype=np.float64)
    if img.ndim > 2:
        img = img.mean(axis=2)
    g = np.gradient(img)
    energy = g[0] ** 2 + g[1] ** 2
    aniso = float(energy.std() / (energy.mean() + 1e-12))
    inc = np.diff(img, axis=1).ravel()
    inc = inc[np.isfinite(inc)]
    if inc.size > 10 and inc.std() > 0:
        inc = inc - inc.mean()
        m2 = np.mean(inc ** 2)
        flat = np.mean(inc ** 4) / (m2 ** 2) if m2 > 1e-18 else 3.0
    else:
        flat = 3.0
    hist, _ = np.histogram(img.ravel(), bins=32, density=True)
    hist = hist[hist > 0]
    ent = float(-np.sum(hist * np.log(hist + 1e-12)))
    return {"aniso": aniso, "flatness": float(flat), "entropy": ent, "energy_mean": float(energy.mean())}

def cheap_descriptor_pvalues(image, n_simulations=24, seed=None):
    rng = np.random.default_rng(seed)
    small = downsample_for_null(image)
    obs = _cheap_metrics(small)
    nulls = {k: [] for k in obs}
    for _ in range(n_simulations):
        try:
            m = _cheap_metrics(phase_randomized_surrogate(small, rng))
        except Exception:
            continue
        for k, val in m.items():
            if np.isfinite(val):
                nulls[k].append(val)
    pmap = {}
    for k, val in obs.items():
        arr = np.array(nulls[k], dtype=float)
        pmap[k] = 1.0 if arr.size == 0 or not np.isfinite(val) else max(float(np.mean(arr >= val)), 1.0 / (arr.size + 1))
    return pmap

def fdr_decision(p_map, alpha=0.05):
    names = list(p_map.keys())
    rejected, p_crit = benjamini_hochberg([p_map[k] for k in names], alpha=alpha)
    passed = [names[i] for i, ok in enumerate(rejected) if ok]
    return {"method": "benjamini-hochberg", "alpha": alpha, "p_values": p_map, "p_critical": p_crit, "passed_descriptors": passed, "n_tested": len(names), "n_passed": len(passed), "fdr_pass": bool(len(passed) > 0)}

def adaptive_surrogate_null_test(image, observed_score, analyze_fn, n_start=24, n_expand=80, p_lo=0.04, p_hi=0.20, seed=None):
    first = surrogate_null_test(image, observed_score, analyze_fn, n_simulations=n_start, seed=seed)
    first["adaptive"] = {"started_at": n_start, "expanded": False}
    p = float(first.get("p_value") or 1.0)
    if p_lo <= p <= p_hi:
        second = surrogate_null_test(image, observed_score, analyze_fn, n_simulations=n_expand, seed=seed)
        second["adaptive"] = {"started_at": n_start, "expanded": True, "expanded_to": n_expand, "p_first": p}
        return second
    return first
