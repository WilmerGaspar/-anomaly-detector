"""Periodicidad espacial = proxy de orden tipo red / condensa / artefacto.

En materiales, un cristal o un metamaterial deja picos discretos en el
espacio reciproco. En el cielo eso es raro de verdad salvo picos del
instrumento (fringing CCD, spikes de difraccion, muestreo).

Este plugin mide picos sobre el continuo radial del |FFT|^2 y avisa
cuando los angulos caen en los ejes del detector.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def analyze_periodicity(image: np.ndarray, n_peaks: int = 8) -> Dict:
    img = np.asarray(image, dtype=np.float64)
    if img.ndim > 2:
        img = img.mean(axis=2)
    img = img - img.mean()
    win = np.outer(np.hanning(img.shape[0]), np.hanning(img.shape[1]))
    spec = np.abs(np.fft.fftshift(np.fft.fft2(img * win))) ** 2
    h, w = spec.shape
    cy, cx = h // 2, w // 2
    spec[cy, cx] = 0.0

    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_int = np.clip(r.astype(int), 0, min(cy, cx) - 1)
    r_max = int(min(cy, cx) * 0.9)

    if min(h, w) > 256:
        half = 128
        spec = spec[cy - half : cy + half, cx - half : cx + half]
        h, w = spec.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        r_int = np.clip(r.astype(int), 0, min(cy, cx) - 1)
        r_max = int(min(cy, cx) * 0.9)

    radial = np.zeros(r_max)
    counts = np.zeros(r_max)
    flat_r = r_int.ravel()
    flat_s = spec.ravel()
    ok = (flat_r >= 1) & (flat_r < r_max)
    np.add.at(radial, flat_r[ok], flat_s[ok])
    np.add.at(counts, flat_r[ok], 1)
    radial = radial / np.maximum(counts, 1)
    if r_max > 5:
        k = np.array([1, 2, 3, 2, 1], dtype=float)
        k /= k.sum()
        radial = np.convolve(radial, k, mode="same")

    idx = np.clip(r_int, 0, max(r_max - 1, 0))
    continuum = radial[idx] if r_max else np.ones_like(spec)
    continuum = np.where(r_int < r_max, continuum, radial[-1] if r_max else 1.0)
    excess = spec / (continuum + 1e-12)

    mask = (r >= 3) & (r < r_max)
    peaks: List[tuple] = []
    step = max(1, min(h, w) // 128)
    for i in range(2, h - 2, step):
        for j in range(2, w - 2, step):
            if not mask[i, j]:
                continue
            val = excess[i, j]
            if val < 8.0:
                continue
            nb = excess[i - 1 : i + 2, j - 1 : j + 2]
            if val >= nb.max():
                ang = float(np.degrees(np.arctan2(i - cy, j - cx)) % 180.0)
                freq = float(r[i, j])
                peaks.append((val, freq, ang, i, j))

    peaks.sort(reverse=True, key=lambda t: t[0])
    picked = []
    for p in peaks:
        if any(abs(p[1] - q[1]) < 2.0 and abs(((p[2] - q[2] + 90) % 180) - 90) < 8 for q in picked):
            continue
        picked.append(p)
        if len(picked) >= n_peaks:
            break

    if picked:
        prominences = np.array([p[0] for p in picked])
        angles = np.array([p[2] for p in picked])
        freqs = np.array([p[1] for p in picked])
        n_sig = int(np.sum(prominences >= 8.0))
        raw = float(np.clip(np.log10(max(prominences[0], 1.0)) / 2.0, 0.0, 1.0))
        peak_score = raw if n_sig >= 2 else raw * 0.35
        axis_dist = np.minimum(angles % 90.0, 90.0 - (angles % 90.0))
        axis_frac = float(np.mean(axis_dist < 8.0)) if len(angles) else 0.0
        lattice_hint = _lattice_hint(angles)
    else:
        prominences = np.array([])
        angles = np.array([])
        freqs = np.array([])
        peak_score = 0.0
        n_sig = 0
        axis_frac = 0.0
        lattice_hint = "ninguno"

    likely_instrument = bool(n_sig >= 2 and axis_frac >= 0.6)

    return {
        "n_peaks": int(len(picked)),
        "n_significant_peaks": n_sig,
        "top_prominence": float(prominences[0]) if len(prominences) else 0.0,
        "periodicity_score": peak_score if not likely_instrument else peak_score * 0.25,
        "peak_frequencies_px": [float(f) for f in freqs[:n_peaks]],
        "peak_angles_deg": [float(a) for a in angles[:n_peaks]],
        "axis_aligned_fraction": axis_frac,
        "lattice_hint": lattice_hint,
        "likely_instrument_artifact": likely_instrument,
        "complexity_score": float(peak_score),
        "note": (
            "Picos alineados al detector: probable fringing/spikes/muestreo."
            if likely_instrument
            else "Picos sobre el continuo radial del espectro 2D."
        ),
    }


def _lattice_hint(angles: np.ndarray) -> str:
    if len(angles) < 3:
        return "insuficiente"
    diffs = []
    a = np.sort(angles)
    for i in range(len(a)):
        d = (a[(i + 1) % len(a)] - a[i]) % 180.0
        if d > 0:
            diffs.append(d)
    if not diffs:
        return "insuficiente"
    med = float(np.median(diffs))
    if abs(med - 60.0) < 12 or abs(med - 120.0) < 12:
        return "hexagonal-like"
    if abs(med - 90.0) < 12:
        return "square-like"
    return "irregular"
