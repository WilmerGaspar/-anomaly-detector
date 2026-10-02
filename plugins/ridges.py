"""Crestas finas y alargadas (filamentos) medidas contra un subrogado IAAFT.

Para cada escala sigma se calcula el hessiano; un pixel es cresta brillante si
la curvatura principal es negativa y fuerte y la otra es pequeña (forma de
linea, no de punto). El umbral de "fuerte" se fija en el percentil 95 de la
misma medida en subrogados IAAFT de la region (mismo espectro e histograma),
asi que el ruido equivalente da ~5 % de pixeles por construccion.

Se cuentan solo los componentes cuyo esqueleto es largo. Los que nacen junto a
una fuente puntual brillante (curvatura negativa en las dos direcciones) se
separan como posibles picos de difraccion (`spike_fraction`). El descriptor
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


def _blobness(img, sigma, min_iso=0.5):
    """Punto, no linea: curvatura negativa en las DOS direcciones y parecida (l2/l1 >= min_iso).
    Devuelve -l2 normalizada por escala; 0 donde la forma es alargada."""
    from scipy.ndimage import gaussian_filter
    hrr = gaussian_filter(img, sigma, order=(2, 0))
    hcc = gaussian_filter(img, sigma, order=(0, 2))
    hrc = gaussian_filter(img, sigma, order=(1, 1))
    tmp = np.sqrt(((hrr - hcc) / 2.0) ** 2 + hrc ** 2)
    l1 = (hrr + hcc) / 2.0 - tmp
    l2 = (hrr + hcc) / 2.0 + tmp
    iso = np.where(l1 < 0, l2 / np.where(l1 < 0, l1, -1.0), 0.0)
    return np.where((l2 < 0) & (iso >= min_iso), -l2, 0.0) * sigma ** 2


def compact_sources(img, null_imgs, sigmas=(1.0, 2.0, 3.0)):
    """Fuentes puntuales: maximos locales redondos cuya 'blobness' supera el MAXIMO de los
    subrogados IAAFT (mismo espectro e histograma), es decir, que el ruido equivalente no produce."""
    from scipy.ndimage import maximum_filter
    b = np.max([_blobness(img, s) for s in sigmas], axis=0)
    thr = float(max(np.max([_blobness(n, s) for s in sigmas]) for n in null_imgs))
    peaks = (b > thr) & (b >= maximum_filter(b, size=5))
    ys, xs = np.nonzero(peaks)
    return np.stack([ys, xs], axis=1), thr


def local_background(lin, box=32):
    """Fondo local: mediana en bloques box x box, interpolada a resolucion completa.
    Sin esto, en una imagen con emision extendida brillante cualquier zona del grumo
    supera fondo_global + N ruido y se toma por fuente (medido en MIRI: 1163 'fuentes')."""
    from scipy.ndimage import zoom
    h, w = lin.shape
    nb_y, nb_x = max(1, h // box), max(1, w // box)
    crop = lin[: nb_y * box, : nb_x * box] if (h >= box and w >= box) else lin
    if h < box or w < box:
        return np.full_like(lin, np.median(lin))
    med = np.median(crop.reshape(nb_y, box, nb_x, box).transpose(0, 2, 1, 3).reshape(nb_y, nb_x, -1), axis=2)
    bg = zoom(med, (h / nb_y, w / nb_x), order=1)
    out = np.empty_like(lin)
    out[: bg.shape[0], : bg.shape[1]] = bg[:h, :w]
    if bg.shape[0] < h:
        out[bg.shape[0]:, :] = out[bg.shape[0] - 1, :]
    if bg.shape[1] < w:
        out[:, bg.shape[1]:] = out[:, bg.shape[1] - 1:bg.shape[1]]
    return out


def _concentration(resid, y, x, scales=(1.0, 1.5, 2.0, 3.0, 4.0)):
    """Max sobre escalas s de (media en r <= 2s - pedestal) / (media en 2s < r <= 4s - pedestal),
    con pedestal = mediana en 4s < r <= 6s. Gaussiana de anchura s: ~20. Grumo mucho mas ancho
    que s (perfil ~ 1 - r^2/2S^2): ~1.5. Con sumas en vez de medias y sin pedestal, el fondo
    del anillo se comia el contraste y estrellas de 50-90 sigma daban ~2 (se perdian)."""
    h, w = resid.shape
    best = 0.0
    for s in scales:
        R = int(np.ceil(6 * s)) + 1
        y0, y1, x0, x1 = max(0, y - R), min(h, y + R + 1), max(0, x - R), min(w, x + R + 1)
        sub = resid[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        r = np.hypot(yy - y, xx - x)
        ped_px = sub[(r > 4 * s) & (r <= 6 * s)]
        inner_px = sub[r <= 2 * s]
        ring_px = sub[(r > 2 * s) & (r <= 4 * s)]
        if not (ped_px.size and inner_px.size and ring_px.size):
            continue
        ped = float(np.median(ped_px))
        inner = float(inner_px.mean()) - ped
        ring = float(ring_px.mean()) - ped
        if inner > 0:
            best = max(best, inner / max(ring, 1e-3 * inner))
    return best


POINT_CONCENTRATION = 5.0   # gaussiana con su propia escala: ~20; grumo extendido: ~1.5


def compact_sources_linear(lin, sigmas=(1.0, 2.0, 3.0), nsig=10.0, box=32, min_concentration=POINT_CONCENTRATION):
    """Fuentes puntuales en valores lineales: maximo local redondo (l2/l1 >= 0.5) cuyo pico
    supera el FONDO LOCAL en `nsig` veces el ruido (MAD del residuo) y cuya luz esta
    concentrada como la de una PSF (_concentration). Los subrogados no sirven aqui como
    umbral: conservan los pixeles brillantes de las estrellas."""
    from scipy.ndimage import maximum_filter
    resid = lin - local_background(lin, box=box)
    b = np.max([_blobness(lin, s) for s in sigmas], axis=0)
    noise = 1.4826 * float(np.median(np.abs(resid - np.median(resid)))) or float(resid.std()) or 1.0
    peaks = (b > 0) & (b >= maximum_filter(b, size=5)) & (maximum_filter(resid, size=3) > nsig * noise)
    ys, xs = np.nonzero(peaks)
    keep = [(y, x) for y, x in zip(ys, xs) if _concentration(resid, int(y), int(x)) >= min_concentration]
    return np.array(keep, dtype=int).reshape(-1, 2), nsig * noise


def _strength(img, sigmas):
    return np.max([_ridge_strength(img, s) for s in sigmas], axis=0)


def _long_skeleton_fraction(mask, min_length, exclude=None):
    """Fraccion de pixeles de esqueleto en componentes cuyo esqueleto mide >= min_length.
    Con esqueleto (no excentricidad) una red de filamentos que se cruzan cuenta entera.
    `exclude`: componentes que tocan esa mascara (alrededor de fuentes puntuales) no
    cuentan; se devuelven aparte como posibles picos de difraccion."""
    from skimage.measure import label
    from skimage.morphology import skeletonize
    skel = skeletonize(mask)
    lab = label(skel, connectivity=2)
    if lab.max() == 0:
        return (0.0, 0.0) if exclude is not None else 0.0
    sizes = np.bincount(lab.ravel())
    long_ = sizes >= min_length
    long_[0] = False
    if exclude is None:
        return float(sizes[long_].sum()) / mask.size
    touch = np.zeros_like(long_)
    touch[np.unique(lab[exclude & (lab > 0)])] = True
    touch[0] = False
    return float(sizes[long_ & ~touch].sum()) / mask.size, float(sizes[long_ & touch].sum()) / mask.size


def analyze_ridges(image: np.ndarray, sigmas=(1.0, 2.0), n_surrogates: int = 3, seed: int = 0, max_side: int = 192,
                   raw: np.ndarray = None, spike_ratio: float = 5.0, spike_decay: float = 1.3) -> Dict:
    """`raw`: la misma region SIN estirar ni recortar (valores lineales). Sirve para
    distinguir picos de difraccion (el nucleo de la estrella es >> que sus picos) de
    cruces de filamentos (el cruce es ~2x el filamento). Sin `raw` no se separan picos."""
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
    # Picos de difraccion: crestas que salen RADIALMENTE de una fuente puntual cuyo nucleo
    # es al menos `spike_ratio` veces mas brillante (sobre el fondo, valores lineales) que
    # la propia cresta. Un cruce de filamentos es ~2x el filamento y una red que pasa
    # junto a una estrella no es radial: ambos se conservan.
    from skimage.measure import label
    from skimage.morphology import skeletonize
    ridge_mask = s_obs > thr
    lin = None
    if raw is not None:
        lin = np.asarray(raw, dtype=np.float64)
        lin = np.where(np.isfinite(lin), lin, np.nanmedian(lin))
        lin = downsample_for_null(lin, max_side=max_side)
    if lin is not None and lin.std() > 0:
        # En valores lineales: al estirar a [0, 1] una estrella brillante satura y su nucleo
        # se funde con los picos en una meseta que ya no parece un punto.
        src, src_thr = compact_sources_linear(lin)
    else:
        src, src_thr = compact_sources(small, surrs)
    spike_lab = set()
    spike_regions = {}                     # (y, x) de la fuente -> alcance maximo de sus picos (px de `small`)
    if lin is not None and len(src):
        bg = float(np.median(lin))
        lab = label(skeletonize(ridge_mask), connectivity=2)
        radius = max(4.0, 0.025 * min(small.shape))
        yy, xx = np.mgrid[:small.shape[0], :small.shape[1]]
        for y, x in src:
            r2 = (yy - y) ** 2 + (xx - x) ** 2
            touching = set(np.unique(lab[(r2 <= (2 * radius) ** 2) & (lab > 0)]).tolist())
            core = float(lin[max(0, y - 1):y + 2, max(0, x - 1):x + 2].max()) - bg
            for c in touching:
                pix = lab == c
                arm_pix = pix & (r2 > radius ** 2) & (r2 <= (4 * radius) ** 2)
                if not arm_pix.any() or core <= 0:
                    continue
                arm = float(np.median(lin[arm_pix])) - bg
                if core < spike_ratio * max(arm, 1e-12 * core):
                    continue
                ang = np.degrees(np.arctan2(yy[pix] - y, xx[pix] - x)) % 360
                hist = np.sort(np.histogram(ang, bins=72, range=(0, 360))[0])[::-1]
                # Un pico de difraccion se apaga al alejarse de la estrella; un filamento que
                # pasa cerca mantiene su brillo (medido: sin esta prueba, 40 estrellas sobre
                # filamentos hacian que los tramos cercanos se tomaran por picos y se taparan).
                inner = pix & (r2 > radius ** 2) & (r2 <= (2 * radius) ** 2)
                outer = pix & (r2 > (3 * radius) ** 2) & (r2 <= (6 * radius) ** 2)
                decays = (not outer.any() or not inner.any() or
                          float(np.median(lin[inner])) - bg >= spike_decay * max(float(np.median(lin[outer])) - bg, 1e-12))
                if hist[:8].sum() >= 0.6 * hist.sum() and decays:     # pocos rayos, radial y se apaga
                    spike_lab.add(c)
                    ext = float(np.sqrt(r2[pix].max()))
                    spike_regions[(int(y), int(x))] = max(spike_regions.get((int(y), int(x)), 0.0), ext)
        near = np.isin(lab, list(spike_lab)) if spike_lab else np.zeros(small.shape, dtype=bool)
    else:
        near = np.zeros(small.shape, dtype=bool)
    spike_src = len(spike_lab)
    f_obs, f_spike = _long_skeleton_fraction(ridge_mask, min_length, exclude=near)
    f_null = float(np.mean([_long_skeleton_fraction(s > thr, min_length) for s in s_null]))
    return {
        "filament_fraction": float(f_obs),
        "filament_fraction_null": f_null,
        "filament_excess": float(f_obs - f_null),
        "n_compact_sources": int(len(src)),
        "n_spike_components": int(spike_src),
        # Fuentes con picos de difraccion, en coordenadas de la imagen de entrada:
        # (fila, columna, radio que cubre sus picos). discovery.py las enmascara.
        "spike_regions": [[float(y * img.shape[0] / small.shape[0]), float(x * img.shape[1] / small.shape[1]),
                           float(ext * img.shape[0] / small.shape[0])] for (y, x), ext in spike_regions.items()],
        # Pixeles de esqueleto de los picos (coordenadas de `small`) para enmascararlos como bandas finas.
        "spike_pixels": np.argwhere(near).tolist() if near.any() else [],
        "small_shape": [int(small.shape[0]), int(small.shape[1])],
        "spike_fraction": float(f_spike),
        "threshold": thr,
        "min_length_px": float(min_length),
        "analysis_side_px": int(max(small.shape)),
        "n_surrogates": int(n_surrogates),
        "note": "Fracción de píxeles en crestas largas y finas, región menos subrogados IAAFT. Las crestas "
                "que nacen junto a una fuente puntual (posibles picos de difracción) van aparte en spike_fraction.",
    }
