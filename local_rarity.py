"""Mapa local de rareza."""
from __future__ import annotations
import numpy as np
from scipy import stats

def _tiles(image, tile):
    h, w = image.shape[:2]
    for y in range(0, h - tile + 1, tile):
        for x in range(0, w - tile + 1, tile):
            yield y, x, image[y:y+tile, x:x+tile]

def local_kurtosis_map(image, tile=32):
    img = np.asarray(image, dtype=np.float64)
    h, w = img.shape[:2]
    out = np.zeros((h, w), dtype=np.float32)
    for y, x, patch in _tiles(img, tile):
        v = patch.ravel()
        v = v[np.isfinite(v)]
        k = 3.0 if v.size < 16 or v.std() == 0 else float(stats.kurtosis(v, fisher=False))
        out[y:y+tile, x:x+tile] = k
    return out

def local_beta_map(image, tile=64):
    img = np.asarray(image, dtype=np.float64)
    h, w = img.shape[:2]
    out = np.full((h, w), np.nan, dtype=np.float32)
    for y, x, patch in _tiles(img, tile):
        if patch.std() == 0:
            continue
        F = np.fft.fftshift(np.fft.fft2(patch - patch.mean()))
        P = np.abs(F) ** 2
        cy, cx = patch.shape[0] // 2, patch.shape[1] // 2
        yy, xx = np.indices(patch.shape)
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
        ks, ps = [], []
        for k in range(2, min(cy, cx)):
            ring = P[r == k]
            if ring.size:
                ks.append(k); ps.append(float(ring.mean()))
        if len(ks) < 6:
            continue
        slope = np.polyfit(np.log10(ks), np.log10(np.clip(ps, 1e-18, None)), 1)[0]
        out[y:y+tile, x:x+tile] = float(-slope)
    return out

def rarity_heatmap(image, tile=32):
    kurt = local_kurtosis_map(image, tile=tile)
    beta = local_beta_map(image, tile=max(tile * 2, 48))
    k_z = (kurt - 3.0) / (np.nanstd(kurt) + 1e-6)
    heat = np.clip(np.abs(k_z) / 4.0, 0, 1)
    if np.isfinite(beta).any():
        b = np.where(np.isfinite(beta), beta, np.nanmedian(beta))
        heat = np.clip(0.6 * heat + 0.4 * np.abs(b - np.nanmedian(b)) / 3.0, 0, 1)
    yy, xx = np.unravel_index(int(np.nanargmax(heat)), heat.shape)
    return {"heatmap": heat, "kurtosis_map": kurt, "beta_map": beta, "peak_yx": [int(yy), int(xx)], "peak_value": float(heat[yy, xx]), "tile": tile, "note": "Pico = mosaico mas raro. Apuntar espectro ahi."}
