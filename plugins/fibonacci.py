"""Detector de razones tipo Fibonacci. Reporta n_pairs_tested."""
from __future__ import annotations
import numpy as np

PHI = 0.5 * (1.0 + 5.0 ** 0.5)
FIB_RATIOS = (1.6180339887, 2.6180339887)

def _nearest_phi(ratio):
    r = float(ratio)
    if r < 1.0:
        r = 1.0 / r if r > 0 else 0.0
    best = min(FIB_RATIOS, key=lambda t: abs(r - t))
    return best, abs(r - best)

def analyze_fibonacci(image):
    img = np.asarray(image, dtype=np.float64)
    if img.ndim > 2:
        img = img.mean(axis=2)
    empty = {"phi": PHI, "n_phi_pairs": 0, "n_pairs_tested": 0, "best_ratio": 0.0, "phi_error": 1.0, "fibonacci_score": 0.0, "radial_phi_hits": 0, "fft_phi_hits": 0, "note": "sin pares"}
    if img.size < 16:
        empty["note"] = "imagen demasiado pequena"
        return empty
    img = np.nan_to_num(img - np.nanmean(img), nan=0.0)
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
    tested = 0
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
                    tested += 1
                    ratio = float(peaks[j] / peaks[i])
                    target, err = _nearest_phi(ratio)
                    if err < 0.05 and 1.5 <= ratio <= 2.8:
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
        freqs = np.sort(freqs)[:16]
        for i in range(len(freqs)):
            for j in range(i + 1, len(freqs)):
                tested += 1
                ratio = float(freqs[j] / freqs[i])
                target, err = _nearest_phi(ratio)
                if err < 0.04 and 1.52 <= ratio <= 2.75:
                    fft_phi += 1
                    if err < best_err:
                        best_err = err
                        best_ratio = ratio
    hits = radial_phi + fft_phi
    rate = hits / tested if tested else 0.0
    score = float(min(1.0, rate * 8.0 + (0.0 if best_err >= 1 else max(0.0, 1.0 - best_err / 0.05) * 0.2)))
    return {"phi": PHI, "n_phi_pairs": int(hits), "n_pairs_tested": int(tested), "best_ratio": float(best_ratio), "phi_error": float(best_err if best_ratio else 1.0), "fibonacci_score": score, "radial_phi_hits": int(radial_phi), "fft_phi_hits": int(fft_phi), "note": "score = hits/tested; 1.5 ya no cuenta como phi"}
