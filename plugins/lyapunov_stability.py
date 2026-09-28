"""Proxy 1D tipo Rosenstein sobre el perfil medio. No es dinamica 2D."""
from __future__ import annotations
import numpy as np

class LyapunovStability:
    def __init__(self, max_iterations=200, embedding_dim=3):
        self.max_iterations = max_iterations
        self.embedding_dim = embedding_dim

    def analyze(self, image):
        img = np.asarray(image, dtype=np.float64)
        if img.ndim > 2:
            img = img.mean(axis=2)
        series = np.nanmean(img, axis=1)
        series = np.nan_to_num(series - np.nanmean(series), nan=0.0)
        if series.size < 30 or series.std() == 0:
            return self._pack(float("nan"), "not_tested")
        dim = self.embedding_dim
        n = len(series) - dim
        if n < 20:
            return self._pack(float("nan"), "not_tested")
        traj = np.column_stack([series[i:i + n] for i in range(dim)])
        divergences = []
        steps = min(8, n // 4)
        for i in range(0, n - steps, max(1, n // 40)):
            d = np.linalg.norm(traj - traj[i], axis=1)
            d[i] = np.inf
            j = int(np.argmin(d))
            if not np.isfinite(d[j]) or d[j] <= 0:
                continue
            for k in range(1, steps):
                if i + k >= n or j + k >= n:
                    break
                sep = float(np.linalg.norm(traj[i + k] - traj[j + k]))
                if sep > 0:
                    divergences.append((k, np.log(sep / (d[j] + 1e-12))))
        if len(divergences) < 8:
            return self._pack(float("nan"), "not_tested")
        by = {}
        for k, val in divergences:
            by.setdefault(k, []).append(val)
        ks = sorted(by)
        ys = [float(np.mean(by[k])) for k in ks]
        if len(ks) < 3:
            return self._pack(float("nan"), "not_tested")
        slope = float(np.polyfit(ks[: min(5, len(ks))], ys[: min(5, len(ys))], 1)[0])
        return self._pack(slope, "not_tested")

    def _pack(self, lam, p):
        return {
            "max_lyapunov": lam,
            "dynamics_type": "proxy 1D exploratorio",
            "p": p,
            "note": "Rosenstein sobre el perfil fila-media. No implica caos del campo 2D.",
        }
