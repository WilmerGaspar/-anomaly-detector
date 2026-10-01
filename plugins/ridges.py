"""Crestas finas y alargadas (filamentos) medidas contra un subrogado IAAFT.

Para cada escala sigma se calcula el hessiano; un pixel es cresta brillante si
la curvatura principal es negativa y fuerte y la otra es pequeña (forma de
linea, no de punto). El umbral de "fuerte" se fija en el percentil 95 de la
misma medida en subrogados IAAFT de la region (mismo espectro e histograma),
asi que el ruido equivalente da ~5 % de pixeles por construccion.

Se cuentan solo los componentes cuyo esqueleto es largo. El descriptor
`filament_excess` es la fraccion de pixeles en esos componentes en la region
menos la de los subrogados: ~0 en ruido, positivo con filamentos.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def _ridge_strength(img, sigma):
    from scipy.ndimage import gaussian_filter
    hrr = gaussian_filter(img, sigma, order=(2, 0))
    hcc = gaussian_filter(img, sigma, order=(0, 2))
    hrc = gaussian_filter(img, sigma, order=(1, 1))
    tmp = np.sqrt(((hrr - hcc) / 2.0) ** 2 + hrc ** 2)
    l1 = (hrr + hcc) / 2.0 - tmp           # curvatura mas negativa
    l2 = (hrr + hcc) / 2.0 + tmp
    line = np.clip(1.0 - np.abs(l2) / (np.abs(l1) + 1e-12), 0.0, 1.0)
    return np.where(l1 < 0, -l1 * line, 0.0) * sigma ** 2      # normalizada por escala


def _strength(img, sigmas):
    return np.max([_ridge_strength(img, s) for s in sigmas], axis=0)


def _long_skeleton_fraction(mask, min_length):
    """Fraccion de pixeles de esqueleto en componentes cuyo esqueleto mide >= min_length.
    Con esqueleto (no excentricidad) una red de filamentos que se cruzan cuenta entera."""
    from skimage.measure import label
    from skimage.morphology import skeletonize
    skel = skeletonize(mask)
    lab = label(skel, connectivity=2)
    if lab.max() == 0:
        return 0.0
    sizes = np.bincount(lab.ravel())[1:]
    return float(sizes[sizes >= min_length].sum()) / mask.size


def analyze_ridges(image: np.ndarray, sigmas=(1.0, 2.0), n_surrogates: int = 3, seed: int = 0, max_side: int = 192) -> Dict:
    from scoring import downsample_for_null, iaaft_surrogate

    img = np.asarray(image, dtype=np.float64)
    if img.ndim > 2:
        img = img.mean(axis=2)
    small = downsample_for_null(img, max_side=max_side)
    if small.std() == 0 or min(small.shape) < 32:
        return {"filament_excess": 0.0, "filament_fraction": 0.0, "filament_fraction_null": 0.0,
                "note": "región uniforme o demasiado pequeña"}
    rng = np.random.default_rng(seed)
    surrs = [iaaft_surrogate(small, rng) for _ in range(int(n_surrogates))]
    s_obs = _strength(small, sigmas)
    s_null = [_strength(s, sigmas) for s in surrs]
    thr = float(np.percentile(np.concatenate([s.ravel() for s in s_null]), 95))
    min_length = max(12.0, 0.1 * min(small.shape))
    f_obs = _long_skeleton_fraction(s_obs > thr, min_length)
    f_null = float(np.mean([_long_skeleton_fraction(s > thr, min_length) for s in s_null]))
    return {
        "filament_fraction": float(f_obs),
        "filament_fraction_null": f_null,
        "filament_excess": float(f_obs - f_null),
        "threshold": thr,
        "min_length_px": float(min_length),
        "analysis_side_px": int(max(small.shape)),
        "n_surrogates": int(n_surrogates),
        "note": "Fracción de píxeles en crestas largas y finas, región menos subrogados IAAFT.",
    }
