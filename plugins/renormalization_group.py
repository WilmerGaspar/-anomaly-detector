"""Longitud de correlacion por ACF. Sin np.random."""
from __future__ import annotations
import numpy as np

class RenormalizationGroup:
    def __init__(self, num_iterations=5):
        self.num_iterations = num_iterations

    def analyze(self, image):
        img = np.asarray(image, dtype=np.float64)
        if img.ndim > 2:
            img = img.mean(axis=2)
        if max(img.shape) > 128:
            step = max(1, max(img.shape) // 128)
            img = img[::step, ::step]
        img = img - np.nanmean(img)
        img = np.nan_to_num(img, nan=0.0)
        if img.std() == 0:
            return self._pack(float("nan"), False, 0.0, "not_tested")
        max_lag = min(100, min(img.shape) // 2)
        acf = []
        for lag in range(1, max_lag + 1):
            a = img[:, :-lag] if img.shape[1] > lag else None
            b = img[:, lag:] if img.shape[1] > lag else None
            c = img[:-lag, :] if img.shape[0] > lag else None
            d = img[lag:, :] if img.shape[0] > lag else None
            vals = []
            if a is not None and a.size:
                vals.append(float(np.corrcoef(a.ravel(), b.ravel())[0, 1]))
            if c is not None and c.size:
                vals.append(float(np.corrcoef(c.ravel(), d.ravel())[0, 1]))
            vals = [v for v in vals if np.isfinite(v)]
            acf.append(float(np.mean(vals)) if vals else 0.0)
        xi = float(max_lag)
        clipped = True
        for lag, val in enumerate(acf, start=1):
            if val <= 1.0 / np.e:
                xi = float(lag)
                clipped = False
                break
        # scale invariance: coarse 2x vs 4x variance ratio closeness to 1
        c2 = img[::2, ::2]
        c4 = img[::4, ::4]
        v2 = float(c2.var()) if c2.size else 0.0
        v4 = float(c4.var()) if c4.size else 0.0
        if v2 > 0 and v4 > 0:
            scale_inv = float(np.clip(1.0 - abs(np.log10((v2 + 1e-12) / (v4 + 1e-12))) / 2.0, 0.0, 1.0))
        else:
            scale_inv = float("nan")
        crit = float(np.clip(xi / float(max_lag), 0.0, 1.0))
        return self._pack(xi, clipped, scale_inv, "not_tested", crit, max_lag)

    def _pack(self, xi, clipped, scale_inv, p, crit=0.0, max_lag=100):
        return {
            "correlation_length": xi,
            "correlation_length_clipped": bool(clipped),
            "correlation_length_max_lag": int(max_lag),
            "scale_invariance_score": scale_inv,
            "criticality_score": crit,
            "p": p,
            "note": "xi = lag ACF<=1/e. clipped si no cruza antes del lag max.",
        }
