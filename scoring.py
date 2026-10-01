"""Score global, nulos por subrogados, FDR y nulos adaptativos.

Cambios respecto a la version anterior (ver CHANGES.md):
- Fases de subrogado tomadas de la FFT de ruido blanco real: respetan la
  simetria hermitica, asi el espectro de potencia se conserva exactamente.
- Nulo por defecto = IAAFT (Schreiber & Schmitz 1996): conserva espectro E
  histograma. El nulo solo de fase volvia gaussiano el histograma y rechazaba
  por no-gaussianidad de un punto, no por estructura espacial.
- p-valor de Monte Carlo estandar: (1 + #{nulo >= obs}) / (n + 1).
- El score observado se calcula a la MISMA resolucion que los nulos.
- Descriptores del FDR: solo estadisticos que dependen de la fase. Se quitaron
  'entropy' (histograma, identico bajo IAAFT) y 'energy_mean' (fijado por el
  espectro, identico bajo cualquier subrogado de fase).
"""
from __future__ import annotations

import numpy as np

DEFAULT_N_NULL = 99          # p minimo = 0.01
FDR_SCOPE = ("aniso", "flatness_lag1", "flatness_lag4", "incr_skew")

FEATURE_GETTERS = {
    "fractal_base": lambda r: r.get("complexity_score", np.nan),
    "kolmogorov_1941": lambda r: 0.6 * r.get("intermittency_factor", np.nan) + 0.4 * r.get("turbulence_intensity", np.nan),
    "lyapunov_stability": lambda r: r.get("chaos_strength", np.nan),
    "persistent_homology": lambda r: r.get("complexity_score", np.nan),
    "renormalization_group": lambda r: r.get("criticality_score", np.nan),
    "anisotropy": lambda r: r.get("anisotropy_index", np.nan),
    "periodicity": lambda r: r.get("periodicity_score", np.nan),
    "entropy": lambda r: float(np.clip(float(r.get("mutual_information_approx", np.nan)) / 4.0, 0.0, 1.0)),
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
            names.append(key)
            vals.append(v)
    return names, np.array(vals, dtype=float)


def compute_global_score(plugin_results, mode=None):
    """`mode` se acepta por compatibilidad; ya no altera el score."""
    _, vals = extract_feature_vector(plugin_results)
    if len(vals) == 0:
        return float("nan")
    return float(np.clip(np.mean(np.clip(vals, 0.0, 1.0)), 0.0, 1.0))


def mc_p_value(null, observed):
    """p-valor de Monte Carlo de una cola (North et al. 2002)."""
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(observed):
        return 1.0
    return float((1 + np.sum(null >= observed)) / (null.size + 1))


# ---------------------------------------------------------------- subrogados

def _random_phases(shape, rng):
    """Fases con simetria hermitica: angulo de la rfft2 de ruido blanco real."""
    return np.angle(np.fft.rfft2(rng.standard_normal(shape)))


def phase_randomized_surrogate(image, rng):
    """Subrogado de fase: conserva |FFT| exactamente; el histograma tiende a gaussiano."""
    img = np.asarray(image, dtype=np.float64)
    f = np.fft.rfft2(img)
    phase = _random_phases(img.shape, rng)
    phase[0, 0] = np.angle(f[0, 0])          # conserva el signo de la media
    return np.fft.irfft2(np.abs(f) * np.exp(1j * phase), s=img.shape)


def iaaft_surrogate(image, rng, n_iter=30):
    """IAAFT: conserva el histograma exacto y el espectro de forma aproximada.
    Termina cuando el orden de los pixeles deja de cambiar (punto fijo)."""
    img = np.asarray(image, dtype=np.float64)
    target_mag = np.abs(np.fft.rfft2(img))
    sorted_vals = np.sort(img.ravel())
    surr = rng.permutation(img.ravel()).reshape(img.shape)
    prev_order = None
    for _ in range(n_iter):
        f = np.fft.rfft2(surr)
        spec = np.fft.irfft2(target_mag * np.exp(1j * np.angle(f)), s=img.shape)
        order = np.argsort(spec.ravel())
        flat = np.empty_like(sorted_vals)
        flat[order] = sorted_vals
        surr = flat.reshape(img.shape)
        if prev_order is not None and np.array_equal(order, prev_order):
            break
        prev_order = order
    return surr


SURROGATES = {"iaaft": iaaft_surrogate, "phase": phase_randomized_surrogate}


def downsample_for_null(image, max_side=96):
    img = np.asarray(image, dtype=np.float64)
    h, w = img.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return img
    try:
        from skimage.transform import resize
        scale = max_side / side
        return resize(img, (max(16, int(h * scale)), max(16, int(w * scale))),
                      anti_aliasing=True, preserve_range=True).astype(np.float64)
    except Exception:
        step = max(1, side // max_side)
        return img[::step, ::step]


def surrogate_null_test(image, observed_score=None, analyze_fn=None, n_simulations=DEFAULT_N_NULL,
                        seed=None, method="iaaft"):
    """Test de subrogados. El score observado se recalcula sobre la imagen reducida
    para que observado y nulo esten a la misma resolucion. `observed_score` se
    acepta por compatibilidad y se guarda solo como referencia."""
    if analyze_fn is None:
        raise ValueError("analyze_fn es obligatorio")
    rng = np.random.default_rng(seed)
    small = downsample_for_null(image)
    observed = float(analyze_fn(small))
    make = SURROGATES[method]
    null_scores = []
    for _ in range(int(n_simulations)):
        try:
            null_scores.append(float(analyze_fn(make(small, rng))))
        except Exception:
            continue
    base = {"method": "%s surrogates" % method, "seed": seed, "observed_score": observed,
            "observed_score_full_res": observed_score, "analysis_side_px": int(max(small.shape))}
    if not null_scores:
        base.update({"p_value": 1.0, "stars": "ns", "is_significant": False, "n_simulations": 0,
                     "null_scores": [], "z_score": 0.0})
        return base
    null = np.array(null_scores, dtype=float)
    p_value = mc_p_value(null, observed)
    null_mean = float(null.mean())
    null_std = float(null.std(ddof=1)) if len(null) > 1 else 0.0
    z = (observed - null_mean) / null_std if null_std > 0 else 0.0
    stars = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    base.update({"p_value": p_value, "p_min_possible": 1.0 / (len(null) + 1), "stars": stars,
                 "is_significant": p_value < 0.05, "null_mean": null_mean, "null_std": null_std,
                 "n_simulations": int(len(null)), "z_score": float(z), "null_scores": null.tolist()})
    return base


# ---------------------------------------------------------------- descriptores baratos

def _crop_to_blocks(img, bs):
    h, w = img.shape
    return img[: (h // bs) * bs, : (w // bs) * bs]


def _flatness(x):
    x = x[np.isfinite(x)]
    if x.size < 10:
        return 3.0
    x = x - x.mean()
    m2 = np.mean(x ** 2)
    return float(np.mean(x ** 4) / m2 ** 2) if m2 > 1e-18 else 3.0


def cheap_score_from_image(image):
    img = np.asarray(image, dtype=np.float64)
    if img.ndim > 2:
        img = img.mean(axis=2)
    g = np.gradient(img)
    energy = g[0] ** 2 + g[1] ** 2
    aniso = float(np.clip((energy.std() / (energy.mean() + 1e-12)) / 3.0, 0.0, 1.0))
    bs = max(4, min(img.shape) // 8)
    blk = _crop_to_blocks(img, bs)                      # sin bloques parciales en el borde
    m = blk.reshape(blk.shape[0] // bs, bs, blk.shape[1] // bs, bs).sum(axis=(1, 3)).ravel()
    lac = float(np.clip((m.var() / ((m.mean() ** 2) + 1e-12)) / 4.0, 0.0, 1.0)) if m.size else 0.0
    inter = float(np.clip((_flatness(np.diff(img, axis=1).ravel()) - 3.0) / 6.0, 0.0, 1.0))
    return float(np.clip(0.4 * aniso + 0.3 * lac + 0.3 * inter, 0.0, 1.0))


def _cheap_metrics(image):
    """Estadisticos sensibles a la fase (no fijados por espectro ni histograma)."""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim > 2:
        img = img.mean(axis=2)
    g = np.gradient(img)
    energy = g[0] ** 2 + g[1] ** 2
    aniso = float(energy.std() / (energy.mean() + 1e-12))
    d1 = np.concatenate([np.diff(img, axis=1).ravel(), np.diff(img, axis=0).ravel()])
    d4 = np.concatenate([(img[:, 4:] - img[:, :-4]).ravel(), (img[4:, :] - img[:-4, :]).ravel()])
    s = d1.std()
    skew = float(abs(np.mean((d1 - d1.mean()) ** 3)) / s ** 3) if s > 0 else 0.0
    return {"aniso": aniso, "flatness_lag1": _flatness(d1), "flatness_lag4": _flatness(d4), "incr_skew": skew}


def cheap_descriptor_null_details(image, n_simulations=DEFAULT_N_NULL, seed=None, method="iaaft"):
    """Como cheap_descriptor_pvalues, pero devuelve por descriptor el valor
    observado, la distribucion nula completa, p, z y la banda 5-95 % del nulo."""
    rng = np.random.default_rng(seed)
    small = downsample_for_null(image)
    obs = _cheap_metrics(small)
    make = SURROGATES[method]
    nulls = {k: [] for k in obs}
    for _ in range(int(n_simulations)):
        try:
            m = _cheap_metrics(make(small, rng))
        except Exception:
            continue
        for k, val in m.items():
            if np.isfinite(val):
                nulls[k].append(val)
    out = {}
    for k, val in obs.items():
        null = np.asarray(nulls[k], dtype=float)
        sd = float(null.std(ddof=1)) if null.size > 1 else 0.0
        mean = float(null.mean()) if null.size else float("nan")
        out[k] = {"observed": float(val), "null": null.tolist(), "p": mc_p_value(null, val),
                  "null_mean": mean, "null_std": sd, "z": float((val - mean) / sd) if sd > 0 else 0.0,
                  "null_q05": float(np.percentile(null, 5)) if null.size else float("nan"),
                  "null_q95": float(np.percentile(null, 95)) if null.size else float("nan")}
    return out


def cheap_descriptor_pvalues(image, n_simulations=DEFAULT_N_NULL, seed=None, method="iaaft"):
    details = cheap_descriptor_null_details(image, n_simulations=n_simulations, seed=seed, method=method)
    return {k: d["p"] for k, d in details.items()}


def benjamini_hochberg(p_values, alpha=0.05):
    p = np.asarray(p_values, dtype=float)
    p = p[np.isfinite(p)]
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool), float("nan")
    order = np.argsort(p)
    ranked = p[order]
    below = ranked <= (np.arange(1, m + 1) / m) * alpha
    if not below.any():
        return np.zeros(m, dtype=bool), float("nan")
    k_max = int(np.max(np.where(below)[0]))
    rejected = np.zeros(m, dtype=bool)
    rejected[order[: k_max + 1]] = True
    return rejected, float(ranked[k_max])


def fdr_decision(p_map, alpha=0.05, n_simulations=None):
    names = [k for k in FDR_SCOPE if k in p_map] or list(p_map.keys())
    rejected, p_crit = benjamini_hochberg([p_map[k] for k in names], alpha=alpha)
    passed = [names[i] for i, ok in enumerate(rejected) if ok]
    out = {
        "method": "benjamini-hochberg",
        "alpha": alpha,
        "scope": names,
        "note": "FDR sobre estadisticos sensibles a la fase con nulo IAAFT. Plugins ricos = no testeados.",
        "p_values": p_map,
        "p_critical": p_crit,
        "passed_descriptors": passed,
        "n_tested": len(names),
        "n_passed": len(passed),
        "fdr_pass": bool(passed),
    }
    if n_simulations:
        p_min = 1.0 / (n_simulations + 1)
        out["p_min_possible"] = p_min
        out["can_pass"] = bool(p_min <= alpha / max(1, len(names)))
    return out


def adaptive_surrogate_null_test(image, observed_score=None, analyze_fn=None, n_start=49, n_expand=199,
                                 p_lo=0.02, p_hi=0.20, seed=None, method="iaaft"):
    first = surrogate_null_test(image, observed_score, analyze_fn, n_simulations=n_start, seed=seed, method=method)
    first["adaptive"] = {"started_at": n_start, "expanded": False}
    p = float(first.get("p_value") or 1.0)
    if p_lo <= p <= p_hi:
        second = surrogate_null_test(image, observed_score, analyze_fn, n_simulations=n_expand, seed=seed, method=method)
        second["adaptive"] = {"started_at": n_start, "expanded": True, "expanded_to": n_expand, "p_first": p}
        return second
    return first
