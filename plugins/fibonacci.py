"""Detector de razones tipo Fibonacci / numero aureo en escalas de la imagen.

No afirma filotaxis cosmica. Mide si picos de FFT o anillos radiales
se agrupan cerca de phi = (1+sqrt(5))/2 o de F_{n+1}/F_n.
"""
from __future__ import annotations

import numpy as np

PHI = 0.5 * (1.0 + 5.0 ** 0.5)
FIB_RATIOS = (1.0, 1.5, 1.6180339887, 1.6666667, 2.0, 2.5, 2.6180339887)


def _nearest_phi(ratio: float) -> tuple:
    r = float(ratio)
    if r < 1.0:
        r = 1.0 / r if r > 0 else 0.0
    best = min(FIB_RATIOS, key=lambda t: abs(r - t))
    return best, abs(r - best)


def analyze_fibonacci(image):
    img = np.asarray(image, dtype=np.float64)
    if img.ndim > 2:
        img = img.mean(axis=2)
    if img.size < 16:
        return {"phi": PHI, "n_phi_pairs": 0, "best_ratio": 0.0, "phi_error": 1.0, "fibonacci_score": 0.0, "radial_phi_hits": 0, "fft_phi_hits": 0, "note": "imagen demasiado pequena"}
    img = img - np.nanmean(img)
    img = np.nan_to_num(img, nan=0.0)
    ny, nx = img.shape
    win = np.outer(np.hanning(ny), np.hanning(nx))
    spec = np.abs(np.fft.fftshift(np.fft.fft2(img * win)))
    cy, cx = ny // 2, nx // 2
    spec[cy, cx] = 0.0
    yy, xx = np.ogrid[:ny, :nx]
    rr = np.hypot(yy - cy, xx - cx)
    r_max = int(min(cy, cx) * 0.9)
    radial = []
    for r in range(2, max(r_max, 3)):
        ring = spec[(rr >= r - 0.5) & (rr < r + 0.5)]
        if ring.size:
            radial.append((r, float(ring.mean())))
    radial_phi = 0
    best_ratio = 0.0
    best_err = 1.0
    if len(radial) >= 8:
        vals = np.array([v for _, v in radial], dtype=np.float64)
        radii = np.array([r for r, _ in radial], dtype=np.float64)
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) + 1e-12
        peaks = radii[vals > med + 2.0 * mad]
        if peaks.size >= 2:
            peaks = np.unique(np.round(peaks, 0))
            for i in range(len(peaks)):
                for j in range(i + 1, len(peaks)):
                    ratio = float(peaks[j] / peaks[i])
                    target, err = _nearest_phi(ratio)
                    if err < 0.08 and 1.4 <= ratio <= 2.8:
                        radial_phi += 1
                        if err < best_err:
                            best_err = err
                            best_ratio = ratio
    flat = spec.ravel()
    thr = float(np.percentile(flat, 99.7))
    ys, xs = np.nonzero(spec >= thr)
    freqs = np.hypot(ys - cy, xs - cx)
    freqs = np.unique(np.round(freqs[freqs > 2.0], 1))
    fft_phi = 0
    if freqs.size >= 2:
        freqs = np.sort(freqs)[:24]
        for i in range(len(freqs)):
            for j in range(i + 1, len(freqs)):
                ratio = float(freqs[j] / freqs[i])
                target, err = _nearest_phi(ratio)
                if err < 0.06 and 1.45 <= ratio <= 2.75:
                    fft_phi += 1
                    if err < best_err:
                        best_err = err
                        best_ratio = ratio
    hits = radial_phi + fft_phi
    score = float(min(1.0, 0.15 * hits + (0.0 if best_err >= 1 else max(0.0, 1.0 - best_err / 0.08) * 0.4)))
    return {"phi": PHI, "n_phi_pairs": int(hits), "best_ratio": float(best_ratio), "phi_error": float(best_err if best_ratio else 1.0), "fibonacci_score": score, "radial_phi_hits": int(radial_phi), "fft_phi_hits": int(fft_phi), "note": "razones ~ phi en FFT/anillos; no implica filotaxis"}
