"""Kolmogorov 1941/62: P(k), beta con error, kurtosis de incrementos, isotropia real."""
from __future__ import annotations

import numpy as np
from scipy import fft, stats


def k62_intermittency(image):
    img = np.asarray(image, dtype=np.float64)
    if img.ndim > 2:
        img = img.mean(axis=2)
    inc = np.concatenate([np.diff(img, axis=1).ravel(), np.diff(img, axis=0).ravel()])
    inc = inc[np.isfinite(inc)]
    if inc.size < 100 or inc.std() == 0:
        return float("nan")
    return float(stats.kurtosis(inc, fisher=False))


def isotropic_score(image):
    img = np.asarray(image, dtype=np.float64)
    if img.ndim > 2:
        img = img.mean(axis=2)
    ny, nx = img.shape
    win = np.outer(np.hanning(ny), np.hanning(nx))
    P = np.abs(np.fft.fftshift(np.fft.fft2(img * win))) ** 2
    ky = np.fft.fftshift(np.fft.fftfreq(ny) * ny)[:, None]
    kx = np.fft.fftshift(np.fft.fftfreq(nx) * nx)[None, :]
    k_mag = np.sqrt(kx ** 2 + ky ** 2)
    cvs = []
    for k0 in range(5, max(6, min(ny, nx) // 4)):
        ring = (k_mag >= k0) & (k_mag < k0 + 1)
        if int(ring.sum()) < 20:
            continue
        vals = P[ring]
        mu = float(vals.mean())
        if mu > 0:
            cvs.append(float(vals.std() / mu))
    if not cvs:
        return float("nan")
    return float(np.clip(1.0 - np.mean(cvs), 0.0, 1.0))


def fit_beta_with_error(k, Pr):
    k = np.asarray(k, dtype=float)
    Pr = np.asarray(Pr, dtype=float)
    mask = (k > 0) & np.isfinite(Pr) & (Pr > 0)
    if int(mask.sum()) < 8:
        return {"beta": 1.67, "beta_se": float("nan"), "r_squared": 0.0, "k_range": [float("nan"), float("nan")], "n_k_bins": int(mask.sum())}
    log_k = np.log10(k[mask])
    log_P = np.log10(Pr[mask])
    n = len(log_k)
    lo, hi = n // 10, n - n // 10
    if hi - lo < 5:
        lo, hi = 0, n
    slope, _intercept, r, se, _p = stats.linregress(log_k[lo:hi], log_P[lo:hi])
    k_ok = k[mask]
    return {"beta": float(-slope), "beta_se": float(se), "r_squared": float(r ** 2), "k_range": [float(k_ok[lo]), float(k_ok[min(hi, len(k_ok) - 1)])], "n_k_bins": int(hi - lo)}


class Kolmogorov1941:
    def __init__(self, num_shells=20):
        self.num_shells = num_shells

    def analyze(self, image):
        img = np.asarray(image, dtype=np.float64)
        if img.ndim > 2:
            img = img.mean(axis=2)
        power = np.abs(fft.fftshift(fft.fft2(img))) ** 2
        k_values, spectrum_radial = self._radial_average(power)
        fit = fit_beta_with_error(k_values, spectrum_radial)
        kurt = k62_intermittency(img)
        iso = isotropic_score(img)
        inter = float(np.clip((kurt - 3.0) / 6.0, 0.0, 1.0)) if np.isfinite(kurt) else 0.0
        return {
            "k_values": k_values.tolist()[:50] if len(k_values) else [],
            "spectrum": spectrum_radial.tolist()[:50] if len(spectrum_radial) else [],
            "beta": float(np.clip(fit["beta"], 0.0, 6.0)),
            "beta_se": fit["beta_se"],
            "r_squared": float(np.clip(fit["r_squared"], 0.0, 1.0)),
            "k_range": fit["k_range"],
            "n_k_bins": fit["n_k_bins"],
            "k62_kurtosis": kurt,
            "intermittency_factor": inter,
            "turbulence_intensity": float(np.clip(1.0 - inter * 0.5, 0.0, 1.0)),
            "integral_scale": float(np.mean(img.shape) / 4.0),
            "isotropic_score": iso if np.isfinite(iso) else 0.0,
        }

    def _radial_average(self, power_spectrum):
        h, w = power_spectrum.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
        r_max = min(cy, cx)
        k_values, spectrum_values = [], []
        for radius in range(1, min(r_max, self.num_shells * 5)):
            ring = (r >= radius - 0.5) & (r < radius + 0.5)
            if np.any(ring):
                avg = float(np.mean(power_spectrum[ring]))
                if avg > 0 and np.isfinite(avg):
                    k_values.append(float(radius))
                    spectrum_values.append(avg)
        return np.array(k_values), np.array(spectrum_values)
