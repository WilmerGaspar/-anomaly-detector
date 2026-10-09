"""Kolmogorov 1941/62: P(k) con bins log-k y SE honesto."""
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

def fit_beta_with_error(k, Pr, n_bins=12):
    k = np.asarray(k, dtype=float)
    Pr = np.asarray(Pr, dtype=float)
    mask = (k > 0) & np.isfinite(Pr) & (Pr > 0)
    if int(mask.sum()) < 8:
        return {"beta": float("nan"), "beta_se": float("nan"), "r_squared": 0.0, "k_range": [float("nan"), float("nan")], "n_k_bins": int(mask.sum())}
    log_k = np.log10(k[mask])
    log_P = np.log10(Pr[mask])
    lo, hi = float(log_k.min()), float(log_k.max())
    if hi <= lo:
        return {"beta": float("nan"), "beta_se": float("nan"), "r_squared": 0.0, "k_range": [float("nan"), float("nan")], "n_k_bins": 0}
    edges = np.linspace(lo, hi, n_bins + 1)
    xs, ys = [], []
    for i in range(n_bins):
        sel = (log_k >= edges[i]) & (log_k < edges[i + 1] if i < n_bins - 1 else log_k <= edges[i + 1])
        if int(sel.sum()) < 2:
            continue
        xs.append(float(log_k[sel].mean()))
        ys.append(float(log_P[sel].mean()))
    # linregress devuelve (pendiente, ordenada, r, p, error): antes se desempaquetaba como
    # (pendiente, ordenada, r, error, p) y "beta_se" era el valor p del ajuste. Casi siempre
    # salia < 1e-4 y se convertia en NaN (null en 13 de 14 JSON reales); en el otro era el p.
    if len(xs) < 5:
        fit = stats.linregress(log_k, log_P)
        slope, r, se = fit.slope, fit.rvalue, fit.stderr
        beta_se = float(se)
        if not np.isfinite(beta_se) or beta_se < 1e-4:
            beta_se = float("nan")
        return {"beta": float(-slope), "beta_se": beta_se, "r_squared": float(r ** 2), "k_range": [float(k[mask].min()), float(k[mask].max())], "n_k_bins": int(mask.sum())}
    fit = stats.linregress(xs, ys)
    slope, r, se = fit.slope, fit.rvalue, fit.stderr
    beta_se = float(se)
    if not np.isfinite(beta_se) or beta_se < 1e-4:
        beta_se = float("nan")
    return {
        "beta": float(-slope),
        "beta_se": beta_se,
        "r_squared": float(r ** 2),
        "k_range": [float(10 ** xs[0]), float(10 ** xs[-1])],
        "n_k_bins": int(len(xs)),
    }

class Kolmogorov1941:
    def __init__(self, num_shells=20, nu_c=None):
        self.num_shells = num_shells
        self.nu_c = nu_c            # corte de la difraccion (ciclos/pixel); ver psf.py

    def analyze(self, image):
        img = np.asarray(image, dtype=np.float64)
        if img.ndim > 2:
            img = img.mean(axis=2)
        # Sin media y con ventana de Hanning, como el espectro de la grafica. Sin ventana, los
        # bordes de la region (no periodica) meten una componente ~k^-3 que arrastra beta hacia 3:
        # medido en recortes de campos de beta conocido, 3.67 -> 3.05; y en un MIRI F1000W real
        # (NGC 7023) daba 2.45 frente a 1.66 con ventana, por el gradiente de brillo.
        h, w = img.shape
        win = np.outer(np.hanning(h), np.hanning(w))
        power = np.abs(fft.fftshift(fft.fft2((img - img.mean()) * win))) ** 2
        k_values, spectrum_radial = self._radial_average(power)
        fit = fit_beta_with_error(k_values, spectrum_radial)
        psf = None
        if self.nu_c:
            # Misma correccion que el espectro de la grafica: P / MTF^2 hasta X_MAX * nu_c.
            from psf import MIN_RANGE, X_MAX, mtf
            side = float(min(h, w))
            keep = k_values / side <= X_MAX * self.nu_c
            raw = fit
            if keep.sum() >= 8 and k_values[keep].max() / max(k_values[keep].min(), 1e-9) >= MIN_RANGE:
                m2 = np.maximum(mtf(k_values[keep] / side, self.nu_c) ** 2, 1e-6)
                fit = fit_beta_with_error(k_values[keep], spectrum_radial[keep] / m2)
            else:
                fit = {"beta": float("nan"), "beta_se": float("nan"), "r_squared": 0.0,
                       "k_range": [float("nan"), float("nan")], "n_k_bins": int(keep.sum())}
            psf = {"beta_raw": raw["beta"], "corrected": True}
        kurt = k62_intermittency(img)
        iso = isotropic_score(img)
        clipped = False
        if np.isfinite(kurt):
            raw_inter = (kurt - 3.0) / 6.0
            clipped = raw_inter > 1.0 or raw_inter < 0.0
            inter = float(np.clip(raw_inter, 0.0, 1.0))
        else:
            inter = float("nan")
        beta = fit["beta"]
        if np.isfinite(beta):
            beta = float(np.clip(beta, 0.0, 6.0))
        return {
            "k_values": k_values.tolist()[:50] if len(k_values) else [],
            "spectrum": spectrum_radial.tolist()[:50] if len(spectrum_radial) else [],
            "beta": beta,
            "beta_se": fit["beta_se"],
            "r_squared": float(np.clip(fit["r_squared"], 0.0, 1.0)),
            "k_range": fit["k_range"],
            "n_k_bins": fit["n_k_bins"],
            "k62_kurtosis": kurt,
            "intermittency_factor": inter,
            "intermittency_clipped": clipped,
            "turbulence_intensity": float(np.clip(1.0 - (inter if np.isfinite(inter) else 0.0) * 0.5, 0.0, 1.0)),
            "isotropic_score": iso if np.isfinite(iso) else float("nan"),
            "beta_raw": psf["beta_raw"] if psf else beta,
            "psf_corrected": bool(psf),
        }

    def _radial_average(self, power_spectrum):
        h, w = power_spectrum.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        # Anillos centrados en cada radio (|r - radio| < 0.5). Antes r se truncaba a entero y el
        # anillo "radio" cubria [radio, radio+1): k medio radio+0.5 asignado a radio, beta ~3 % bajo.
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
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
