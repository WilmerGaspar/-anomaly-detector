"""Controles de discovery.py (semáforo), golden.py (ventana phi) y radio_signals.py.
Las cifras de referencia se midieron antes de fijar umbrales (CHANGES.md 52-60)."""
import numpy as np
import pytest

import radio_signals as rs
from discovery import evaluate, fdr_without_point_sources, mask_point_sources, novelty_vs_references
from golden import GOLDEN_ANGLE, PHI, golden_angle_test, phi_scale_test


def _field(n, beta, seed):
    rng = np.random.default_rng(seed)
    k = np.hypot(np.fft.fftfreq(n)[:, None], np.fft.fftfreq(n)[None, :])
    k[0, 0] = 1
    f = np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n))) * k ** (-beta / 2)).real
    return f / f.std()


def _stars_only(seed, n=384, k=30):
    rng = np.random.default_rng(seed)
    g = _field(n, 3.0, seed)
    yy, xx = np.mgrid[:n, :n]
    for _ in range(k):
        cy, cx = rng.uniform(10, n - 10, 2)
        g = g + rng.uniform(5, 60) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.8 ** 2))
    return g


# ---------------------------------------------------------------- semaforo

def test_point_sources_masked_removes_signal():
    raw = _stars_only(2)
    filled, info = mask_point_sources(raw)
    assert info["n_masked"] >= 15 and info["masked_fraction"] < 0.05
    assert fdr_without_point_sources(raw, n_simulations=49)["n_passed"] == 0


def _card(**over):
    card = {"source": {"provenance": {"verdict": "usable_science"}},
            "analysis": {"empty_area_fraction": 0.0},
            "morphology": {"instrument_warning": False, "n_point_sources": 0},
            "structure_test": {"fdr": {"fdr_pass": True, "n_passed": 3, "n_tested": 4}},
            "descriptors": {"ridges": {"filament_excess": 0.0, "n_spike_components": 0}}}
    for k, v in over.items():
        card[k].update(v)
    return card


REP = {"fdr_pass": True, "n_passed": 3, "seed": 1}
MASK_OK = {"fdr_pass": True, "n_passed": 3, "n_masked": 0, "masked_fraction": 0.0}


def test_untested_controls_never_count_in_favour():
    assert evaluate(_card(morphology={"n_point_sources": 93}))["level"] == "unconfirmed"
    # Sin la prueba de enmascarado no hay verde, aunque el detector a 10 sigma no vea fuentes.
    assert evaluate(_card(), replicate=REP)["level"] == "unconfirmed"


def test_levels():
    assert evaluate(_card(), masked=MASK_OK, replicate=REP)["level"] == "robust"
    assert evaluate(_card(analysis={"empty_area_fraction": 0.2}), masked=MASK_OK, replicate=REP)["level"] == "invalid"
    masked_fail = {"fdr_pass": False, "n_passed": 0, "n_masked": 93, "masked_fraction": 0.02}
    assert evaluate(_card(morphology={"n_point_sources": 93}), masked=masked_fail, replicate=REP)["level"] == "explained"
    spikes = _card(descriptors={"ridges": {"filament_excess": 0.0, "n_spike_components": 2}},
                   morphology={"n_point_sources": 1})
    assert evaluate(spikes, masked=dict(masked_fail, n_masked=1), replicate=REP)["level"] == "explained"
    assert evaluate(spikes, masked=dict(MASK_OK, n_masked=1), replicate=REP)["level"] == "robust"
    none = _card(structure_test={"fdr": {"fdr_pass": False, "n_passed": 0, "n_tested": 4}})
    assert evaluate(none, masked=MASK_OK, replicate=REP)["level"] == "none"


def test_novelty_needs_references_and_flags_outlier():
    refs = [_card(descriptors={"fractal_base": {"d0": 1.7 + 0.01 * i}}) for i in range(6)]
    assert not novelty_vs_references(refs[0], refs[:3])["available"]
    odd = _card(descriptors={"fractal_base": {"d0": 1.2}})
    nv = novelty_vs_references(odd, refs)
    assert nv["available"] and nv["max_abs_z"] > 5
    assert evaluate(odd, masked=MASK_OK, replicate=REP, references=refs)["level"] == "pioneer"


def _one_star(n, cy, cx, amp=400, spike_amp=40, length=120, sig=2.5):
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(xx - cx, yy - cy)
    g = amp * np.exp(-r ** 2 / (2 * sig ** 2))
    for ang in np.deg2rad([113, 173, 53]):
        d = np.abs(np.cos(ang) * (xx - cx) + np.sin(ang) * (yy - cy))
        g = g + spike_amp * np.exp(-d ** 2 / 2) * np.exp(-r / length)
    return g


@pytest.mark.parametrize("seed", [0, 3])
def test_single_bright_star_with_spikes_is_explained(seed):
    from discovery import stretch01
    from plugins.ridges import analyze_ridges
    img = _field(600, 3.0, seed) + _one_star(600, 180, 420)
    rid = analyze_ridges(stretch01(img), raw=img)
    assert rid["n_spike_components"] >= 1
    m = fdr_without_point_sources(img, spike_pixels=rid["spike_pixels"], small_shape=rid["small_shape"])
    assert m["n_passed"] == 0 and m["masked_fraction"] < 0.08     # bandas finas, no discos


# ---------------------------------------------------------------- ventana phi

def test_golden_angle_vogel_vs_random():
    rng = np.random.default_rng(0)
    k = np.arange(1, 60)
    th = np.deg2rad(k * GOLDEN_ANGLE)
    vog = np.c_[8 * np.sqrt(k) * np.sin(th), 8 * np.sqrt(k) * np.cos(th)] + rng.normal(0, 0.8, (59, 2))
    assert golden_angle_test(vog, n_perm=499)["p"] < 0.005
    assert golden_angle_test(rng.uniform(-60, 60, (59, 2)), n_perm=499)["p"] > 0.05


@pytest.mark.parametrize("seed", [0, 1])
def test_phi_rings_detected_but_not_3_2_rings(seed):
    n = 1024
    rr = np.hypot(*np.mgrid[-n // 2:n // 2, -n // 2:n // 2])
    g = _field(n, 2.5, seed)
    phi = g + 0.6 * sum(np.cos(2 * np.pi * k * rr) for k in (0.05, 0.05 * PHI, 0.05 * PHI ** 2))
    other = g + 0.6 * sum(np.cos(2 * np.pi * k * rr) for k in (0.05, 0.075, 0.1125))
    assert phi_scale_test(phi, n_null=500, seed=seed)["alert"]
    assert not phi_scale_test(other, n_null=500, seed=seed)["alert"]


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_phi_constant_image_does_not_crash(value):
    # Una extension ERR uniforme hacia caer la app (espectro todo cero -> array vacio).
    r = phi_scale_test(np.full((300, 300), value))
    assert r["p"] is None and r["alert"] is False and "variación" in r["note"]


# ---------------------------------------------------------------- radio

def test_periodicity_noise_and_pulsar():
    fa = sum(rs.periodicity_search(np.random.default_rng(s).standard_normal(2 ** 13), 1e-3)["detected"] for s in range(60))
    assert fa <= 3
    red = sum(rs.periodicity_search(np.cumsum(np.random.default_rng(s).standard_normal(2 ** 13)) * 0.05
                                    + np.random.default_rng(s + 9).standard_normal(2 ** 13), 1e-3)["detected"] for s in range(20))
    assert red <= 2
    r = rs.periodicity_search(rs.gen_pulsar(2 ** 14, 1e-3, 0.0893, amp=0.15, seed=3), 1e-3)
    assert r["detected"] and min(abs(r["best_period_s"] * h / 0.0893 - 1) for h in (1, 2, 0.5, 3, 1 / 3, 4, 0.25)) < 0.01


def test_dispersion_burst_and_rfi():
    dyn, f = rs.gen_dispersed_burst(nchan=32, ntime=1024, dm=500, amp=0.7, seed=1)
    r = rs.dispersion_search(dyn, f, 1e-3, n_dm=100, n_null=9)
    assert r["detected"] and abs(r["dm"] - 500) < 25 and not r["likely_rfi"]
    imp = rs.dispersion_search(rs.gen_rfi_impulse(nchan=32, ntime=1024, seed=2), f, 1e-3, n_dm=100, n_null=9)
    assert imp["detected"] and imp["likely_rfi"]
    noise = rs.dispersion_search(np.random.default_rng(7).standard_normal((32, 1024)), f, 1e-3, n_dm=100, n_null=9)
    assert not noise["detected"]


def test_drift_tone_and_fixed_rfi():
    r = rs.drift_search(rs.gen_drifting_tone(nchan=256, drift=-0.15, amp=0.6, f_start_chan=150, seed=1), 3.0, 10.0, n_null=9)
    assert r["detected"] and abs(r["drift_hz_s"] + 0.15) < 0.01 and not r["likely_rfi"]
    fixed = rs.drift_search(rs.gen_drifting_tone(nchan=256, drift=0.0, amp=0.8, f_start_chan=100, seed=2), 3.0, 10.0, n_null=9)
    assert fixed["detected"] and fixed["likely_rfi"]
    noise = rs.drift_search(np.random.default_rng(3).standard_normal((64, 256)), 3.0, 10.0, n_null=9)
    assert not noise["detected"]
    assert not r["range_limited"] and r["drift_range_hz_s"] >= 1.0


def test_drift_budget_keeps_step_and_says_range(monkeypatch):
    # 1024 x 1024 con ±1 Hz/s tardaba ~25 min. Con el tope: mismo paso, rango menor y avisado.
    # Aqui el tope se reduce para que la prueba sea rapida: 64 x 256 x 10 barridos x 41 derivas.
    monkeypatch.setattr(rs, "DRIFT_WORK_MAX", 64 * 256 * 10 * 41)
    inside = rs.drift_search(rs.gen_drifting_tone(nchan=256, drift=0.05, amp=0.6, f_start_chan=100, seed=1), 3.0, 10.0, n_null=9)
    step = 3.0 / (64 * 10.0)
    assert inside["range_limited"] and inside["drift_step_hz_s"] == step
    assert abs(inside["drift_range_hz_s"] - 20 * step) < 1e-12
    assert inside["detected"] and abs(inside["drift_hz_s"] - 0.05) < 0.01
    outside = rs.drift_search(rs.gen_drifting_tone(nchan=256, drift=-0.15, amp=0.6, f_start_chan=150, seed=1), 3.0, 10.0, n_null=9)
    assert not outside["detected"] and "Rango buscado" in outside["note"]


def test_crossmatch_with_harmonics():
    rows = [{"NAME": "A", "P0": "0.0893", "DM": "68"}, {"NAME": "C", "P0": "1.2", "DM": "300"}]
    m = rs.crossmatch_catalog(rows, period_s=0.0893 / 2)
    assert [x["row"]["NAME"] for x in m] == ["A"] and m[0]["harmonic"] == 2
    assert rs.crossmatch_catalog(rows, period_s=0.5) == []


def test_period_uncertainty_is_honest_and_used_in_crossmatch():
    errs = []
    for s in range(4):
        r = rs.periodicity_search(rs.gen_pulsar(2 ** 14, 1e-3, 0.7145, amp=0.3, seed=s), 1e-3)
        assert r["detected"]
        errs.append(abs(r["best_period_s"] - 0.7145) / r["period_err_s"])
    assert max(errs) < 1.0
    rows = [{"NAME": "A", "P0": "0.7145"}]
    # Antes del afinado salia 0.7123 s (0.3 % de error): con el 0.2 % fijo no se encontraba.
    assert not rs.crossmatch_catalog(rows, period_s=0.7123)
    assert rs.crossmatch_catalog(rows, period_s=0.7123, period_err_s=0.0023)


# ---------------------------------------------------------------- regresion: MIRI F2550W real (CHANGES 67-71)

def test_golden_angle_ignores_clustered_points():
    # Puntos agrupados en grumos: los giros entre vecinos se concentran cerca de 0 grados.
    # El estadístico anterior (|media exp(i(dθ-g))|) daba p = 0.001 aquí.
    rng = np.random.default_rng(1)
    centers = rng.uniform(-200, 200, (12, 2))
    pts = np.vstack([c + rng.normal(0, 6, (25, 2)) for c in centers])
    assert golden_angle_test(pts, n_perm=499)["p"] > 0.05


def _lognormal(seed, n=480):
    g = _field(n, 3.0, seed)
    return np.exp(1.2 * g) * 100 + 900 + np.random.default_rng(seed).normal(0, 5, (n, n))


def test_extended_emission_cannot_make_green():
    from discovery import MAX_MASK_FRACTION
    img = _lognormal(1)
    filled, info = mask_point_sources(img)
    assert all(r < 38 for r in info["radii"])            # nada con el perfil sin cerrar
    if info["masked_fraction"] > MAX_MASK_FRACTION:
        assert not info["valid"]
    card = _card(morphology={"n_point_sources": 50})
    bad = {"fdr_pass": True, "n_passed": 4, "n_masked": 600, "masked_fraction": 0.25, "valid": False}
    assert evaluate(card, masked=bad, replicate=REP)["level"] == "unconfirmed"


def test_stars_detected_against_local_background():
    from discovery import find_point_sources
    assert len(find_point_sources(_stars_only(2), nsig=5)[0]) >= 28          # 30 reales
    assert len(find_point_sources(_field(384, 3.0, 5), nsig=5)[0]) == 0      # ruido: ninguna
