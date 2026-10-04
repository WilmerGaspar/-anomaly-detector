"""Analitica detallada de una region: espectro, escalas, mapa local y tabla completa.

Todo se compara contra subrogados IAAFT (mismo espectro e histograma) para que
cada grafico diga cuanto se aparta la region del ruido equivalente, no solo
cuanto vale un descriptor.
"""
from __future__ import annotations

import numpy as np

from scoring import _cheap_metrics, _flatness, downsample_for_null, iaaft_surrogate


# ---------------------------------------------------------------- espectro

def radial_power_spectrum(image, n_bins=None, window=True):
    """P(k) promediado en anillos. k en ciclos/pixel (0, 0.5].
    window=True (Hanning) reduce las fugas del borde para ajustar beta. Para comparar con un
    subrogado IAAFT hay que usar window=False: IAAFT copia el espectro sin ventana, y con
    ventana las dos curvas se separan ~2x a k bajo aunque el nulo sea correcto (medido)."""
    img = np.asarray(image, dtype=np.float64)
    img = img - img.mean()
    h, w = img.shape
    win = np.outer(np.hanning(h), np.hanning(w)) if window else 1.0
    power = np.abs(np.fft.fftshift(np.fft.fft2(img * win))) ** 2
    ky = np.fft.fftshift(np.fft.fftfreq(h))
    kx = np.fft.fftshift(np.fft.fftfreq(w))
    kr = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    n_bins = n_bins or max(8, min(h, w) // 4)
    edges = np.linspace(0, 0.5, n_bins + 1)
    idx = np.digitize(kr.ravel(), edges) - 1
    k, p = [], []
    for i in range(n_bins):
        sel = idx == i
        if i == 0 or not sel.any():                          # sin el modo k=0
            continue
        k.append(0.5 * (edges[i] + edges[i + 1]))
        p.append(float(power.ravel()[sel].mean()))
    return np.array(k), np.array(p)


def fit_power_law(k, p, k_min=0.02, k_max=0.25):
    """Ajuste log-log P(k) ~ k^-beta en el rango inercial. Devuelve beta, R^2 y la recta."""
    k, p = np.asarray(k), np.asarray(p)
    sel = (k >= k_min) & (k <= k_max) & (p > 0)
    if sel.sum() < 3:
        return {"beta": float("nan"), "r2": float("nan"), "k_min": k_min, "k_max": k_max, "n_points": int(sel.sum())}
    x, y = np.log10(k[sel]), np.log10(p[sel])
    slope, icpt = np.polyfit(x, y, 1)
    pred = slope * x + icpt
    ss_res, ss_tot = np.sum((y - pred) ** 2), np.sum((y - y.mean()) ** 2)
    return {"beta": float(-slope), "intercept": float(icpt), "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "k_min": k_min, "k_max": k_max, "n_points": int(sel.sum())}


# ---------------------------------------------------------------- escalas

def _increments(img, lag):
    return np.concatenate([(img[:, lag:] - img[:, :-lag]).ravel(), (img[lag:, :] - img[:-lag, :]).ravel()])


def scale_profile(image, lags=(1, 2, 4, 8, 16), n_surrogates=19, seed=None, max_side=128):
    """Curtosis de incrementos y funcion de estructura S2 por escala, observado vs banda IAAFT.
    Una curtosis > 3 que el nulo no reproduce indica intermitencia (bordes, filamentos)."""
    rng = np.random.default_rng(seed)
    small = downsample_for_null(image, max_side=max_side)
    lags = [int(l) for l in lags if l < min(small.shape) // 2]

    def prof(img):
        flat, s2 = [], []
        for l in lags:
            d = _increments(img, l)
            flat.append(_flatness(d))
            s2.append(float(np.mean(d ** 2)))
        return flat, s2

    obs_f, obs_s2 = prof(small)
    nf, ns = [], []
    for _ in range(int(n_surrogates)):
        f, s2 = prof(iaaft_surrogate(small, rng))
        nf.append(f)
        ns.append(s2)
    nf, ns = np.array(nf), np.array(ns)
    return {"lags": lags, "flatness": obs_f, "s2": obs_s2,
            "flatness_null_mean": nf.mean(axis=0).tolist(), "flatness_null_q05": np.percentile(nf, 5, axis=0).tolist(),
            "flatness_null_q95": np.percentile(nf, 95, axis=0).tolist(),
            "s2_null_mean": ns.mean(axis=0).tolist(), "n_surrogates": int(n_surrogates), "analysis_side_px": int(max(small.shape))}


def increment_histograms(image, lag=1, seed=None, bins=60, max_side=256):
    """Histograma normalizado (por su desviacion tipica) de incrementos: region vs un subrogado."""
    rng = np.random.default_rng(seed)
    small = downsample_for_null(image, max_side=max_side)
    surr = iaaft_surrogate(small, rng)
    a, b = _increments(small, lag), _increments(surr, lag)
    a, b = a / (a.std() or 1), b / (b.std() or 1)
    edges = np.linspace(-8, 8, bins + 1)
    ha, _ = np.histogram(a, edges, density=True)
    hb, _ = np.histogram(b, edges, density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    gauss = np.exp(-centers ** 2 / 2) / np.sqrt(2 * np.pi)
    return {"x": centers.tolist(), "observed": ha.tolist(), "surrogate": hb.tolist(), "gaussian": gauss.tolist(), "lag": lag}


# ---------------------------------------------------------------- mapa local

def local_significance_map(image, grid=4, n_surrogates=19, seed=None, tile_side=64, metric="flatness_lag1", valid=None):
    """Divide la region en grid x grid teselas y calcula, en cada una, z del descriptor
    frente a sus propios subrogados IAAFT. Exploratorio: con 19 subrogados z es aproximado
    y no hay correccion por comparaciones multiples. `valid` (mascara de pixeles con dato):
    las teselas con algun pixel vacio quedan en NaN en lugar de evaluarse."""
    rng = np.random.default_rng(seed)
    img = np.asarray(image, dtype=np.float64)
    h, w = img.shape
    th, tw = h // grid, w // grid
    z = np.full((grid, grid), np.nan)
    obs = np.full((grid, grid), np.nan)
    if th < 16 or tw < 16:
        return {"z": z.tolist(), "observed": obs.tolist(), "grid": grid, "metric": metric, "note": "región demasiado pequeña"}
    for i in range(grid):
        for j in range(grid):
            if valid is not None and not np.all(valid[i * th:(i + 1) * th, j * tw:(j + 1) * tw]):
                continue
            tile = downsample_for_null(img[i * th:(i + 1) * th, j * tw:(j + 1) * tw], max_side=tile_side)
            if tile.std() == 0:
                continue
            o = _cheap_metrics(tile)[metric]
            null = np.array([_cheap_metrics(iaaft_surrogate(tile, rng, n_iter=15))[metric] for _ in range(int(n_surrogates))])
            sd = null.std(ddof=1)
            obs[i, j] = o
            z[i, j] = (o - null.mean()) / sd if sd > 0 else 0.0
    return {"z": z.tolist(), "observed": obs.tolist(), "grid": grid, "metric": metric, "tile_px": [int(th), int(tw)],
            "n_surrogates": int(n_surrogates)}


# ---------------------------------------------------------------- utilidades

def bh_curve(p_map, alpha=0.05):
    """Puntos para el grafico de Benjamini-Hochberg: p ordenados vs umbral i/m * alpha."""
    items = sorted(p_map.items(), key=lambda kv: kv[1])
    m = len(items)
    return [{"rank": i + 1, "descriptor": k, "p": float(p), "threshold": (i + 1) / m * alpha} for i, (k, p) in enumerate(items)]


def flatten_numeric(results, prefix=""):
    """Aplana los resultados de los plugins a filas (plugin, clave, valor) solo numericas y escalares."""
    rows = []
    for key, val in (results or {}).items():
        name = "%s.%s" % (prefix, key) if prefix else str(key)
        if isinstance(val, dict):
            rows.extend(flatten_numeric(val, name))
        elif isinstance(val, (bool, np.bool_)):
            continue
        elif isinstance(val, (int, float, np.integer, np.floating)) and np.isfinite(float(val)):
            plugin, _, field = name.partition(".")
            rows.append({"plugin": plugin, "descriptor": field or plugin, "valor": float(val)})
    return rows


def surrogate_preview(image, seed=None, max_side=256):
    small = downsample_for_null(image, max_side=max_side)
    return small, iaaft_surrogate(small, np.random.default_rng(seed))


def full_analytics(image, seed=None, valid=None):
    """Todo el paquete analitico en un dict serializable a JSON."""
    small = downsample_for_null(image, max_side=256)
    k, p = radial_power_spectrum(small)
    surr = iaaft_surrogate(small, np.random.default_rng(seed))
    _, pn = radial_power_spectrum(small, window=False)
    _, ps = radial_power_spectrum(surr, window=False)
    return {"spectrum": {"k": k.tolist(), "p": p.tolist(), "p_nowindow": pn.tolist(), "p_surrogate": ps.tolist(),
                         "fit": fit_power_law(k, p)},
            "scales": scale_profile(image, seed=seed),
            "increments": increment_histograms(image, seed=seed),
            "local_map": local_significance_map(image, seed=seed, valid=valid)}
