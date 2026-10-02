"""Ventana Fibonacci / razon aurea: dos tests con nulo explicito.

1. Angulo aureo (filotaxis): fuentes ordenadas por distancia al centro; si cada una gira
   ~137.508 grados respecto a la anterior (espiral de Vogel, girasol). Nulo: barajar el
   orden radial manteniendo las posiciones.
2. Razones phi entre escalas: picos del espectro radial (sobre una ley de potencia
   ajustada) cuyas razones caen cerca de phi o phi^2 mas veces que el mismo numero de
   picos en frecuencias al azar de la misma rejilla.

Alerta solo con p < ALPHA (Bonferroni para 2 tests al 1 %). Ningun resultado de aqui
identifica un mecanismo: phi aparece en sistemas de crecimiento y empaquetamiento,
pero un p pequeno solo dice "no es azar de este tipo".
"""
from __future__ import annotations

import numpy as np

PHI = (1 + 5 ** 0.5) / 2
GOLDEN_ANGLE = 360.0 * (1 - 1 / PHI)           # 137.5077...
ALPHA = 0.01 / 2


def golden_angle_test(points, n_perm=999, seed=0, min_points=10):
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    out = {"test": "ángulo áureo", "n_points": int(len(pts)), "golden_angle_deg": GOLDEN_ANGLE}
    if len(pts) < min_points:
        out.update({"p": None, "alert": False, "note": "Hacen falta al menos %d fuentes." % min_points})
        return out
    c = pts.mean(axis=0)
    d = pts - c
    r = np.hypot(d[:, 0], d[:, 1])
    th = np.arctan2(d[:, 0], d[:, 1])

    def stat(order):
        # Media de cos(dtheta -+ angulo aureo): 1 si cada giro es 137.5 grados, ~0 al azar,
        # negativo si los giros se concentran en otro angulo. La version anterior usaba
        # |media de exp(i(dtheta - g))|, que vale lo mismo para CUALQUIER angulo concentrado
        # (el desfase no cambia el modulo): puntos agrupados en una nube daban p = 0.001.
        dth = np.diff(th[order])
        return max(float(np.mean(np.cos(dth - s * np.deg2rad(GOLDEN_ANGLE)))) for s in (1, -1))

    obs = stat(np.argsort(r))
    rng = np.random.default_rng(seed)
    null = np.array([stat(rng.permutation(len(pts))) for _ in range(int(n_perm))])
    p = float((1 + np.sum(null >= obs)) / (len(null) + 1))
    out.update({"statistic": float(obs), "null_mean": float(null.mean()), "null_q95": float(np.percentile(null, 95)),
                "null": null.tolist(), "p": p, "alert": p < ALPHA,
                "note": "Media de cos(giro − 137.5°) entre fuentes consecutivas por radio: 1 = filotaxis perfecta, "
                        "≈ 0 = azar."})
    return out


def _peaks_over_powerlaw(k, p, nmad=3.0):
    from analytics import fit_power_law
    fit = fit_power_law(k, p, k_min=float(k.min()), k_max=float(k.max()))
    if not np.isfinite(fit.get("beta", np.nan)):
        return np.array([]), fit
    resid = np.log10(p) - (fit["intercept"] - fit["beta"] * np.log10(k))
    med = np.median(resid)
    mad = 1.4826 * np.median(np.abs(resid - med)) or 1e-9
    is_peak = np.zeros(len(k), dtype=bool)
    is_peak[1:-1] = (resid[1:-1] > resid[:-2]) & (resid[1:-1] >= resid[2:]) & (resid[1:-1] - med > nmad * mad)
    return k[is_peak], fit


MAX_PAIR_TOL = 0.05      # phi frente a 3/2 o 7/4 difiere ~7 %: con mas tolerancia no se distingue


def _phi_pairs(ks, tol, dk=0.0):
    """(pares con razon phi o phi^2, pares resolubles). Tolerancia de cada par =
    max(tol, dk/ki + dk/kj): un pico solo se conoce con la precision del intervalo dk.
    Los pares con tolerancia > MAX_PAIR_TOL no se cuentan: no distinguen phi de 3/2."""
    ks = np.sort(np.asarray(ks, dtype=float))
    n = resolvable = 0
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            t_ij = max(tol, dk / ks[i] + dk / ks[j])
            if t_ij > MAX_PAIR_TOL:
                continue
            resolvable += 1
            ratio = ks[j] / ks[i]
            if any(abs(ratio / t - 1) <= t_ij for t in (PHI, PHI ** 2)):
                n += 1
    return n, resolvable


def phi_scale_test(image, n_null=2000, seed=0, tol=0.03):
    from analytics import radial_power_spectrum
    from scoring import downsample_for_null
    small = downsample_for_null(np.asarray(image, dtype=float), max_side=1024)
    k, p = radial_power_spectrum(small, n_bins=max(16, min(small.shape) // 2))
    ok = p > 0
    k, p = k[ok], p[ok]
    peaks, fit = _peaks_over_powerlaw(k, p)
    out = {"test": "razones φ entre escalas", "n_peaks": int(len(peaks)), "peak_k": peaks.tolist(), "tolerance": tol}
    if len(peaks) < 2:
        out.update({"p": None, "alert": False, "n_phi_pairs": 0,
                    "note": "Menos de 2 picos sobre la ley de potencia: no hay escalas que comparar."})
        return out
    dk = float(np.median(np.diff(k)))
    obs, resolvable = _phi_pairs(peaks, tol, dk)
    out["n_resolvable_pairs"] = int(resolvable)
    if resolvable == 0:
        out.update({"p": None, "alert": False, "n_phi_pairs": 0,
                    "note": "Resolución insuficiente: ningún par de picos se conoce con precisión mejor que "
                            "±%d %%, que hace falta para distinguir φ de 3/2. Usa una región más grande." % int(100 * MAX_PAIR_TOL)})
        return out
    rng = np.random.default_rng(seed)
    # Nulo: los mismos picos en posiciones al azar de la rejilla de frecuencias real.
    null = np.array([_phi_pairs(rng.choice(k, size=len(peaks), replace=False), tol, dk)[0] for _ in range(int(n_null))])
    pv = float((1 + np.sum(null >= obs)) / (len(null) + 1))
    out.update({"n_phi_pairs": int(obs), "null_mean": float(null.mean()), "null": null.tolist(), "p": pv,
                "alert": bool(obs > 0 and pv < ALPHA),
                "note": "Pares de picos con razón φ o φ² (tolerancia: la mayor entre ±%d %% y la resolución en "
                        "frecuencia) frente al mismo número de picos en frecuencias al azar." % int(100 * tol)})
    return out


def golden_window(image, points, seed=0):
    a = golden_angle_test(points, seed=seed)
    b = phi_scale_test(image, seed=seed)
    alerts = [t["test"] for t in (a, b) if t.get("alert")]
    return {"tests": [a, b], "alpha": ALPHA, "alert": bool(alerts), "alert_tests": alerts}
