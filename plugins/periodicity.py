"""Periodicidad espacial = proxy de orden tipo red / condensa / artefacto.

En materiales, un cristal o un metamaterial deja picos discretos en el
espacio reciproco. En el cielo eso es raro de verdad salvo picos del
instrumento (fringing CCD, spikes de difraccion, muestreo).

Este plugin mide picos sobre el continuo radial del |FFT|^2 y avisa
cuando los angulos caen en los ejes del detector.

Umbral: en ruido, el exceso sobre el continuo sigue aprox. una exponencial de
media 1, y el maximo de N valores crece como ln N (medido: mediana ~9 en campos
gaussianos de 128-400 px, por encima del antiguo umbral fijo de 8). Ahora el
umbral es ln(N / 0.01): Bonferroni al 1 % sobre los N pixeles independientes
(la mitad del disco examinado, por la simetria del espectro de una imagen real).

Rayas: una linea recta en la imagen deja en la FFT una raya de picos con el
mismo angulo a frecuencias arbitrarias. Eso es un filamento, no una red. Un
grupo de >= 3 picos con el mismo angulo cuyas frecuencias no son armonicos de
la menor se marca como raya y no cuenta como periodicidad.
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
    # Espectro de imagen real: simetrico, solo la mitad de los pixeles es independiente.
    n_tested = max(1, int(mask.sum()) // 2)
    thr = max(8.0, float(np.log(n_tested / 0.01)))
    # Maximos locales 3x3 sobre TODOS los pixeles (antes se submuestreaba y un pico
    # cuyo maximo caia en un pixel saltado se perdia: p. ej. una red girada).
    from scipy.ndimage import maximum_filter
    is_peak = mask & (excess >= thr) & (excess >= maximum_filter(excess, size=3, mode="nearest"))
    peaks: List[tuple] = []
    for i, j in zip(*np.nonzero(is_peak)):
        ang = float(np.degrees(np.arctan2(i - cy, j - cx)) % 180.0)
        peaks.append((float(excess[i, j]), float(r[i, j]), ang, int(i), int(j)))

    peaks.sort(reverse=True, key=lambda t: t[0])
    picked = []
    for p in peaks:
        if any(abs(p[1] - q[1]) < 2.0 and abs(((p[2] - q[2] + 90) % 180) - 90) < 8 for q in picked):
            continue
        picked.append(p)
        if len(picked) >= 3 * n_peaks:      # margen: una raya puede ocupar muchos picos
            break

    picked, streaks = _split_streaks(picked)
    picked = picked[:n_peaks]

    if picked:
        prominences = np.array([p[0] for p in picked])
        angles = np.array([p[2] for p in picked])
        freqs = np.array([p[1] for p in picked])
        n_sig = int(np.sum(prominences >= thr))
        raw = float(np.clip(np.log10(max(prominences[0], 1.0)) / 2.0, 0.0, 1.0))
        lattice_like = _lattice_consistent(freqs[prominences >= thr], angles[prominences >= thr])
        peak_score = raw if (n_sig >= 2 and lattice_like) else raw * 0.35
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
        lattice_like = False

    likely_instrument = bool(n_sig >= 1 and axis_frac >= 0.6)

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
        "lattice_consistent": bool(picked) and bool(lattice_like),
        "peak_threshold": thr,
        "n_tested_pixels": n_tested,
        "streak_angles_deg": [float(a) for a in streaks],
        "n_streaks": int(len(streaks)),
        "complexity_score": float(peak_score),
        "note": (
            "Picos alineados al detector: probable fringing/spikes/muestreo."
            if likely_instrument
            else "Picos sobre el continuo radial del espectro 2D."
        ),
    }


def _lattice_consistent(freqs, angles, rel_tol=0.1, min_sep_deg=15.0):
    """Una red real deja picos a la MISMA frecuencia en direcciones distintas (o armonicos
    en una). Picos sueltos a radios distintos (p. ej. filamentos rectos) no son una red."""
    f, a = np.asarray(freqs, dtype=float), np.asarray(angles, dtype=float)
    for i in range(len(f)):
        for j in range(i + 1, len(f)):
            sep = abs(((a[i] - a[j] + 90) % 180) - 90)
            if sep >= min_sep_deg and abs(f[i] - f[j]) <= rel_tol * max(f[i], f[j]):
                return True
            if sep < min_sep_deg and _is_harmonic([f[i], f[j]]):
                return True
    return False


def _is_harmonic(freqs, tol=0.12):
    f = np.sort(np.asarray(freqs, dtype=float))
    ratio = f / f[0]
    return bool(np.all(np.abs(ratio - np.round(ratio)) <= tol * np.round(ratio)))


def _split_streaks(picked, ang_tol=6.0, min_peaks=3):
    """Separa rayas (lineas rectas en la imagen) de picos discretos.
    Devuelve (picos que quedan, angulos de las rayas)."""
    groups: List[List[tuple]] = []
    for p in picked:
        for g in groups:
            if abs(((p[2] - g[0][2] + 90) % 180) - 90) < ang_tol:
                g.append(p)
                break
        else:
            groups.append([p])
    keep, streaks = [], []
    for g in groups:
        if len(g) >= min_peaks and not _is_harmonic([q[1] for q in g]):
            streaks.append(float(np.median([q[2] for q in g])))
        else:
            keep.extend(g)
    keep.sort(reverse=True, key=lambda t: t[0])
    return keep, streaks


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
