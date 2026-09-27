"""Mascara de spikes de difraccion HST/JWST."""
from __future__ import annotations
import numpy as np

def _peak(image):
    img = np.asarray(image, dtype=np.float64)
    idx = np.unravel_index(np.nanargmax(img), img.shape)
    return int(idx[0]), int(idx[1]), float(img[idx])

def spike_mask(image, n_spikes=6, half_width=2.0, min_radius=4, max_frac=0.48):
    img = np.asarray(image, dtype=np.float64)
    h, w = img.shape[:2]
    cy, cx, peak = _peak(img)
    med = float(np.nanmedian(img))
    if not np.isfinite(peak) or peak < med + 3 * (np.nanstd(img) + 1e-9):
        return np.zeros((h, w), dtype=bool)
    yy, xx = np.indices((h, w))
    dy, dx = yy - cy, xx - cx
    r = np.sqrt(dx * dx + dy * dy)
    ang = np.arctan2(dy, dx)
    rmax = max_frac * min(h, w)
    core = r <= min_radius
    spokes = np.zeros((h, w), dtype=bool)
    bases = [k * np.pi / 3.0 for k in range(n_spikes)] + [np.pi / 2, -np.pi / 2]
    for a0 in bases:
        delta = np.angle(np.exp(1j * (ang - a0)))
        band = np.abs(delta) < (half_width / np.maximum(r, 1.0))
        spokes |= band & (r > min_radius) & (r < rmax)
    return core | spokes

def apply_spike_mask(image, mask=None):
    img = np.asarray(image, dtype=np.float64).copy()
    if mask is None:
        mask = spike_mask(img)
    if mask.any():
        fill = float(np.nanmedian(img[~mask])) if (~mask).any() else 0.0
        img[mask] = fill
    return img, mask

def fft_spoke_mask(shape, n_spikes=6, width=2):
    h, w = shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    dy, dx = yy - cy, xx - cx
    r = np.sqrt(dx * dx + dy * dy)
    ang = np.arctan2(dy, dx)
    spokes = np.zeros((h, w), dtype=bool)
    for k in range(n_spikes):
        a0 = k * np.pi / n_spikes
        delta = np.angle(np.exp(1j * (ang - a0)))
        spokes |= (np.abs(delta) * np.maximum(r, 1.0) < width) & (r > 3)
    return spokes
