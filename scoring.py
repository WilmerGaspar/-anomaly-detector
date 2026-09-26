"""Score global y test nulo por subrogados de fase."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

FEATURE_GETTERS = {
    "fractal_base": lambda r: r.get("complexity_score", np.nan),
    "kolmogorov_1941": lambda r: 0.6 * r.get("intermittency_factor", 0.0)
    + 0.4 * r.get("turbulence_intensity", 0.0),
    "lyapunov_stability": lambda r: r.get("chaos_strength", np.nan),
    "persistent_homology": lambda r: r.get("complexity_score", np.nan),
    "renormalization_group": lambda r: r.get("criticality_score", np.nan),
    "anisotropy": lambda r: r.get("anisotropy_index", np.nan),
    "periodicity": lambda r: r.get("periodicity_score", np.nan),
    "entropy": lambda r: float(
        np.clip(float(r.get("mutual_information_approx", 0.0)) / 4.0, 0.0, 1.0)
    ),
}


def extract_feature_vector(plugin_results: Dict) -> Tuple[List[str], np.ndarray]:
    names, vals = [], []
    for key, getter in FEATURE_GETTERS.items():
        if key not in plugin_results:
            continue
        v = getter(plugin_results[key])
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        names.append(key)
        vals.append(v)
    return names, np.array(vals, dtype=float)


def compute_global_score(plugin_results: Dict, mode: str = "balanced") -> float:
    _, vals = extract_feature_vector(plugin_results)
    if len(vals) == 0:
        score = 0.5
    else:
        score = float(np.clip(np.mean(np.clip(vals, 0.0, 1.0)), 0.0, 1.0))
    if mode == "explorer":
        score = min(score * 1.15, 1.0)
    elif mode == "conservative":
        score = score * 0.85
    return float(np.clip(score, 0.0, 1.0))


def phase_randomized_surrogate(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img = np.asarray(image, dtype=np.float64)
    f = np.fft.rfft2(img)
    mag = np.abs(f)
    rand = rng.uniform(0.0, 2.0 * np.pi, size=f.shape)
    rand[0, 0] = 0.0
    surrogate_f = mag * np.exp(1j * rand)
    surr = np.fft.irfft2(surrogate_f, s=img.shape).real
    if surr.std() > 0 and img.std() > 0:
        surr = (surr - surr.mean()) / surr.std() * img.std() + img.mean()
    return np.clip(surr, float(img.min()), float(img.max()))


def downsample_for_null(image: np.ndarray, max_side: int = 96) -> np.ndarray:
    h, w = image.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return image
    try:
        from skimage.transform import resize
        scale = max_side / side
        nh, nw = max(16, int(h * scale)), max(16, int(w * scale))
        return resize(image, (nh, nw), anti_aliasing=True, preserve_range=True).astype(np.float64)
    except Exception:
        step = max(1, side // max_side)
        return image[::step, ::step]


def surrogate_null_test(
    image: np.ndarray,
    observed_score: float,
    analyze_fn: Callable[[np.ndarray], float],
    n_simulations: int = 40,
    seed: Optional[int] = None,
) -> Dict:
    rng = np.random.default_rng(seed)
    small = downsample_for_null(image)
    null_scores = []
    for _ in range(n_simulations):
        surr = phase_randomized_surrogate(small, rng)
        try:
            null_scores.append(float(analyze_fn(surr)))
        except Exception:
            continue
    if not null_scores:
        return {
            "p_value": 1.0,
            "stars": "ns",
            "is_significant": False,
            "null_mean": float("nan"),
            "null_std": float("nan"),
            "null_se": float("nan"),
            "null_ci95": [float("nan"), float("nan")],
            "n_simulations": 0,
            "method": "phase-randomized surrogates",
            "null_scores": [],
        }
    null = np.array(null_scores, dtype=float)
    p_value = float(np.mean(null >= observed_score))
    p_value = max(p_value, 1.0 / (len(null) + 1))
    null_mean = float(null.mean())
    null_std = float(null.std(ddof=1)) if len(null) > 1 else 0.0
    null_se = float(null_std / np.sqrt(len(null)))
    ci = [null_mean - 1.96 * null_se, null_mean + 1.96 * null_se]
    if p_value < 0.001:
        stars = "***"
    elif p_value < 0.01:
        stars = "**"
    elif p_value < 0.05:
        stars = "*"
    else:
        stars = "ns"
    z = (observed_score - null_mean) / (null_std + 1e-12)
    return {
        "p_value": p_value,
        "stars": stars,
        "is_significant": p_value < 0.05,
        "null_mean": null_mean,
        "null_std": null_std,
        "null_se": null_se,
        "null_ci95": ci,
        "n_simulations": int(len(null)),
        "method": "phase-randomized surrogates (espectro conservado)",
        "z_score": float(z),
        "null_scores": null.tolist(),
    }


def cheap_score_from_image(image: np.ndarray) -> float:
    img = np.asarray(image, dtype=np.float64)
    if img.ndim > 2:
        img = img.mean(axis=2)
    g = np.gradient(img)
    energy = g[0] ** 2 + g[1] ** 2
    aniso = float(np.clip((energy.std() / (energy.mean() + 1e-12)) / 3.0, 0.0, 1.0))
    bs = max(4, min(img.shape) // 8)
    masses = []
    for i in range(0, img.shape[0], bs):
        for j in range(0, img.shape[1], bs):
            masses.append(float(img[i : i + bs, j : j + bs].sum()))
    m = np.array(masses)
    lac = float((m.var() / ((m.mean() ** 2) + 1e-12)) if m.size else 0.0)
    lac = float(np.clip(lac / 4.0, 0.0, 1.0))
    inc = np.diff(img, axis=1).ravel()
    inc = inc - inc.mean()
    m2 = np.mean(inc ** 2)
    m4 = np.mean(inc ** 4)
    flat = m4 / (m2 ** 2) if m2 > 1e-18 else 3.0
    inter = float(np.clip((flat - 3.0) / 6.0, 0.0, 1.0))
    return float(np.clip(0.4 * aniso + 0.3 * lac + 0.3 * inter, 0.0, 1.0))
