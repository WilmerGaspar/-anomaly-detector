"""Controles sinteticos de periodicity.py, ridges.py y la familia 'filament'.
Los valores de referencia se midieron antes de fijar los umbrales (ver CHANGES.md 30-33)."""
import numpy as np
import pytest

from materials_map import family_scores
from plugins.periodicity import analyze_periodicity
from plugins.ridges import analyze_ridges

N = 256
YY, XX = np.mgrid[:N, :N]


def _stretch(a):
    lo, hi = np.percentile(a, [2, 98])
    return np.clip((a - lo) / (hi - lo), 0, 1)


def _field(beta=2.8, seed=0):
    rng = np.random.default_rng(seed)
    k = np.hypot(np.fft.fftfreq(N)[:, None], np.fft.fftfreq(N)[None, :])
    k[0, 0] = 1
    f = np.fft.ifft2(np.fft.fft2(rng.standard_normal((N, N))) * k ** (-beta / 2)).real
    return f / f.std()


def _lines(seed, n=6, parallel=False):
    rng = np.random.default_rng(seed)
    g = _field(seed=seed)
    for _ in range(n):
        a = 0.6 + rng.normal(0, 0.05) if parallel else rng.uniform(0, np.pi)
        d = np.abs(np.cos(a) * XX + np.sin(a) * YY - rng.uniform(0, N))
        g = g + 3 * np.exp(-d ** 2 / 2)
    return g


def _arcs(seed, n=5):
    rng = np.random.default_rng(seed)
    g = _field(seed=seed)
    for _ in range(n):
        cx, cy, r = rng.uniform(0, N, 3)
        d = np.abs(np.hypot(XX - cx, YY - cy) - (r / 2 + 30))
        g = g + 3 * np.exp(-d ** 2 / 2)
    return g


def _grid(angle=0.5, period=10, amp=0.8, seed=0):
    u = XX * np.cos(angle) + YY * np.sin(angle)
    v = -XX * np.sin(angle) + YY * np.cos(angle)
    return _field(seed=seed) + amp * (np.cos(2 * np.pi * u / period) + np.cos(2 * np.pi * v / period))


@pytest.mark.parametrize("beta", [2.0, 2.8, 3.5])
@pytest.mark.parametrize("seed", range(4))
def test_noise_has_no_significant_peaks(beta, seed):
    r = analyze_periodicity(_stretch(_field(beta, seed)))
    assert r["n_significant_peaks"] == 0
    assert r["periodicity_score"] == 0.0


def test_rotated_grid_is_detected():
    # Antes se perdia: el bucle submuestreado saltaba el pixel del maximo.
    r = analyze_periodicity(_stretch(_grid()))
    assert r["n_significant_peaks"] >= 2 and r["periodicity_score"] > 0.5
    assert not r["likely_instrument_artifact"]


def test_axis_fringing_flagged_as_artifact():
    r = analyze_periodicity(_stretch(_field(seed=3) + 0.8 * np.cos(2 * np.pi * XX / 9)))
    assert r["likely_instrument_artifact"]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_straight_lines_are_streaks_not_lattice(seed):
    r = analyze_periodicity(_stretch(_lines(seed, parallel=True)))
    assert r["n_streaks"] >= 1
    assert r["periodicity_score"] < 0.3


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_ridges_separate_filaments_from_noise(seed):
    noise = analyze_ridges(_stretch(_field(seed=seed)))["filament_excess"]
    lines = analyze_ridges(_stretch(_lines(seed)))["filament_excess"]
    arcs = analyze_ridges(_stretch(_arcs(seed)))["filament_excess"]
    grid = analyze_ridges(_stretch(_grid(seed=seed)))["filament_excess"]
    assert noise < 0.003 and grid < 0.003
    assert lines > 0.006 and arcs > 0.006


def _plugins(img):
    from plugins.anisotropy import calculate_anisotropy
    return {"periodicity": analyze_periodicity(img), "ridges": analyze_ridges(img), "anisotropy": calculate_anisotropy(img)}


@pytest.mark.parametrize("seed", [1, 2])
def test_filament_family_beats_lattice_for_lines(seed):
    s = family_scores(_plugins(_stretch(_lines(seed))), structure_z=3.0)
    assert s["filament"] > s["lattice"]


def test_lattice_family_wins_for_grid():
    s = family_scores(_plugins(_stretch(_grid())), structure_z=3.0)
    assert s["lattice"] > s["filament"]
