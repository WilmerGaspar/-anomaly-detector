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


def test_empty_region_is_never_a_candidate():
    from materials_map import interpret
    r = interpret({}, 5.0, 0.01, fdr_pass=True, provenance={"verdict": "usable_science"}, empty_fraction=0.12)
    assert r["state"] == "invalid_region" and not r["is_candidate"]
    ok = interpret({}, 5.0, 0.01, fdr_pass=True, provenance={"verdict": "usable_science"}, empty_fraction=0.0)
    assert ok["state"] != "invalid_region"


def test_local_map_skips_tiles_with_empty_pixels():
    from analytics import local_significance_map
    img = _field(seed=4)
    valid = np.ones_like(img, dtype=bool)
    valid[:20, :20] = False
    z = np.array(local_significance_map(img, grid=2, n_surrogates=3, seed=0, valid=valid)["z"])
    assert np.isnan(z[0, 0]) and np.all(np.isfinite(z.ravel()[1:]))


def _stars(seed, spikes=True, n_src=14):
    rng = np.random.default_rng(seed)
    g = 0.3 * _field(beta=3.3, seed=seed)
    for _ in range(n_src):
        cy, cx = rng.uniform(20, N - 20, 2)
        g = g + rng.uniform(3, 30) * np.exp(-((YY - cy) ** 2 + (XX - cx) ** 2) / (2 * 2.0 ** 2))
    if spikes:                                    # estrella brillante con picos de difraccion
        cy, cx = N * 0.3, N * 0.7
        r = np.hypot(XX - cx, YY - cy)
        for ang in np.deg2rad([113, 173, 53]):
            d = np.abs(np.cos(ang) * (XX - cx) + np.sin(ang) * (YY - cy))
            g = g + 40 * np.exp(-d ** 2 / 2) * np.exp(-r / 60)
        g = g + 400 * np.exp(-((YY - cy) ** 2 + (XX - cx) ** 2) / (2 * 2.5 ** 2))
    return g


def _plugins_raw(raw):
    from plugins.anisotropy import calculate_anisotropy
    img = _stretch(raw)
    return {"periodicity": analyze_periodicity(img), "ridges": analyze_ridges(img, raw=raw),
            "anisotropy": calculate_anisotropy(img)}


@pytest.mark.parametrize("seed", [1, 2])
def test_diffraction_spikes_are_not_filaments(seed):
    raw = _stars(seed)
    r = analyze_ridges(_stretch(raw), raw=raw)
    assert r["filament_excess"] < 0.003 and r["n_spike_components"] >= 1
    s = family_scores(_plugins_raw(raw), structure_z=3.0)
    assert max(s, key=s.get) != "filament"


@pytest.mark.parametrize("seed", [1, 2])
def test_filaments_survive_bright_stars(seed):
    raw = _lines(seed) + _stars(seed, spikes=False, n_src=8) - 0.3 * _field(beta=3.3, seed=seed)
    assert analyze_ridges(_stretch(raw), raw=raw)["filament_excess"] > 0.006


def test_point_sources_flagged_in_verdict():
    from materials_map import interpret
    raw = _stars(3, spikes=False)
    res = {"ridges": analyze_ridges(_stretch(raw), raw=raw)}
    assert res["ridges"]["n_compact_sources"] >= 5
    v = interpret(res, 3.0, 0.01, fdr_pass=True, provenance={"verdict": "usable_science"})
    assert v["point_source_warning"] and "fuentes puntuales" in v["verdict"]
    noise = {"ridges": analyze_ridges(_stretch(_field(seed=3)), raw=_field(seed=3))}
    assert not interpret(noise, 3.0, 0.01, fdr_pass=True, provenance={"verdict": "usable_science"})["point_source_warning"]


def test_isolated_peaks_at_different_radii_are_not_a_lattice():
    from plugins.periodicity import _lattice_consistent
    assert not _lattice_consistent([114.6, 81.1], [9.5, 92.8])          # dos filamentos rectos
    assert _lattice_consistent([34.0, 34.5], [17.0, 77.0])              # red: misma frecuencia
    assert _lattice_consistent([34.0, 67.5], [17.0, 17.0])              # armonicos en una direccion
