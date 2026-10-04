import numpy as np

from analytics import bh_curve, fit_power_law, flatten_numeric, local_significance_map, radial_power_spectrum, scale_profile
from scoring import cheap_descriptor_null_details, cheap_descriptor_pvalues


def _gaussian_field(n=128, beta=3.0, seed=0):
    rng = np.random.default_rng(seed)
    k = np.hypot(np.fft.fftfreq(n)[:, None], np.fft.fftfreq(n)[None, :])
    k[0, 0] = 1
    return np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n))) * k ** (-beta / 2)).real


def test_power_law_slope_recovered():
    k, p = radial_power_spectrum(_gaussian_field(256, beta=3.0))
    fit = fit_power_law(k, p)
    assert abs(fit["beta"] - 3.0) < 0.3
    assert fit["r2"] > 0.95


def test_details_match_pvalues():
    img = _gaussian_field(64)
    det = cheap_descriptor_null_details(img, n_simulations=19, seed=3)
    pv = cheap_descriptor_pvalues(img, n_simulations=19, seed=3)
    assert {k: d["p"] for k, d in det.items()} == pv
    assert all(len(d["null"]) == 19 for d in det.values())


def test_filaments_raise_small_scale_flatness():
    img = _gaussian_field(128, beta=2.8, seed=1)
    yy, xx = np.mgrid[:128, :128]
    for c in (30, 70, 110):
        img = img + 3 * img.std() * np.exp(-((xx + 0.4 * yy - c) ** 2) / 2)
    prof = scale_profile(img, n_surrogates=9, seed=0)
    assert prof["flatness"][0] > prof["flatness_null_q95"][0]


def test_local_map_shape_and_small_region():
    m = local_significance_map(_gaussian_field(128), grid=2, n_surrogates=5, seed=0)
    assert np.array(m["z"]).shape == (2, 2)
    small = local_significance_map(_gaussian_field(32), grid=4, n_surrogates=5, seed=0)
    assert "note" in small


def test_bh_curve_and_flatten():
    curve = bh_curve({"a": 0.04, "b": 0.01}, alpha=0.05)
    assert [r["descriptor"] for r in curve] == ["b", "a"]
    assert curve[1]["threshold"] == 0.05
    rows = flatten_numeric({"p": {"x": 1.0, "y": {"z": 2}, "s": "txt", "flag": True, "nan": float("nan")}})
    assert {(r["plugin"], r["descriptor"]) for r in rows} == {("p", "x"), ("p", "y.z")}


def test_surrogate_spectrum_matches_without_window():
    # IAAFT copia el espectro sin ventana: comparado asi coincide; con ventana Hanning se separaba
    # 2-16x a k bajo y la app decia "el nulo no es valido" sin serlo.
    from analytics import full_analytics
    img = _gaussian_field(128, beta=2.5)
    sp = full_analytics((img - img.min()) / np.ptp(img), seed=1)["spectrum"]
    ratio = np.array(sp["p_surrogate"]) / np.array(sp["p_nowindow"])
    assert np.all(np.abs(np.log10(ratio)) < 0.05)
