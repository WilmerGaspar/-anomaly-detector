"""Difracción del telescopio (psf.py): β corregido y ángulos en el cielo.

Caso real: NGC 7023 con MIRI F2100W daba β 3.72 (≈ 11/3, «Kolmogorov») y con F1000W 1.54. Una
parte grande era la difracción: simulado, F2100W sube un β de 2.5 a 3.6."""
import numpy as np
import pytest

import psf as P


@pytest.mark.parametrize("tel,ins,filt,lam", [
    ("JWST", "MIRI", "F2100W", 21.0), ("JWST", "MIRI", "F770W", 7.7), ("JWST", "NIRCAM", "F200W", 2.0),
    ("JWST", "NIRCAM", "F410M", 4.1), ("HST", "WFC3", "F125W", 1.25), ("HST", "WFC3", "F606W", 0.606),
    ("HST", "ACS", "F814W", 0.814), ("", "MIRI", "F1000W", 10.0), ("", "", "F200W", None), ("JWST", "MIRI", "CLEAR", None)])
def test_wavelength_from_filter_name(tel, ins, filt, lam):
    got = P.wavelength_um(tel, ins, filt)
    assert (got is None and lam is None) or abs(got - lam) < 1e-9


def test_cutoff_and_fwhm_match_jwst():
    d = P.describe({"telescope": "JWST", "instrument": "MIRI", "filter": "F2100W", "pixel_scale_arcsec": 0.11})
    assert abs(d["nu_c_cycles_per_px"] - 0.165) < 0.002
    assert 5.5 < d["fwhm_px"] < 7.0                    # FWHM de MIRI F2100W ≈ 0.67″ ≈ 6 px
    assert P.describe({"telescope": "JWST", "filter": "F2100W"})["nu_c_cycles_per_px"] is None   # sin escala
    assert P.mtf(0.0, 0.2) == pytest.approx(1.0) and P.mtf(0.2, 0.2) == pytest.approx(0.0)


def _diffraction_field(beta, nu_c, seed, n=630, big=2048, noise=0.02):
    rng = np.random.default_rng(seed)
    ky, kx = np.fft.fftfreq(big)[:, None], np.fft.fftfreq(big)[None]
    k = np.hypot(ky, kx)
    kk = np.where(k > 0, k, 1.0)
    f = np.fft.ifft2(np.fft.fft2(rng.normal(size=(big, big))) * kk ** (-beta / 2) * P.mtf(k, nu_c)).real
    c = f[:n, :n] / f[:n, :n].std()
    return c + noise * rng.normal(size=(n, n))


def test_beta_corrected_for_miri_f2100w():
    from analytics import full_analytics
    nu_c = P.cutoff_cycles_per_px("JWST", "MIRI", "F2100W", 0.11)
    fits = [full_analytics(_diffraction_field(2.5, nu_c, s), seed=1, nu_c=nu_c)["spectrum"]["fit"] for s in range(3)]
    assert np.mean([f["beta_raw"] for f in fits]) > 3.3                    # sin corregir: parece «Kolmogorov»
    assert abs(np.mean([f["beta"] for f in fits]) - 2.5) < 0.25
    assert all(f["psf_corrected"] and not f["psf_limited"] for f in fits)


def test_kolmogorov_plugin_uses_the_same_correction():
    from plugins.kolmogorov_1941 import Kolmogorov1941
    nu_c = P.cutoff_cycles_per_px("JWST", "MIRI", "F2100W", 0.11)
    rs = [Kolmogorov1941(nu_c=nu_c).analyze(_diffraction_field(2.5, nu_c, s)) for s in range(3)]
    assert np.mean([r["beta_raw"] for r in rs]) > 3.0
    assert abs(np.mean([r["beta"] for r in rs]) - 2.5) < 0.3 and all(r["psf_corrected"] for r in rs)


def test_beta_not_measurable_when_diffraction_leaves_no_scales():
    from analytics import full_analytics
    nu_c = P.cutoff_cycles_per_px("JWST", "MIRI", "F2550W", 0.11)
    fit = full_analytics(_diffraction_field(2.5, nu_c, 0, n=200, big=1024), seed=1, nu_c=nu_c)["spectrum"]["fit"]
    assert fit["psf_limited"] and not np.isfinite(fit["beta"])
    from hypotheses import NOT_MEASURABLE, build
    card = {"analytics": {"spectrum": {"fit": fit}, "scales": {"flatness": [30, 30], "flatness_null_q95": [9, 9]}},
            "descriptors": {}, "structure_test": {"fdr": {}}, "source": {}}
    rows = {r["key"]: r for r in build(card, "robust")["rows"]}
    assert rows["k41"]["status"] == rows["supersonic"]["status"] == rows["edges"]["status"] == NOT_MEASURABLE
    assert "difracción" in rows["k41"]["why"]


def _wcs(rot_deg, ra=315.4, dec=68.16, pix=0.11):
    """Cabecera WCS TAN con el norte girado `rot_deg` (CD como las de JWST)."""
    t = np.radians(rot_deg)
    s = pix / 3600.0
    return {"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CRVAL1": ra, "CRVAL2": dec, "CRPIX1": 500.0, "CRPIX2": 500.0,
            "CD1_1": -s * np.cos(t), "CD1_2": s * np.sin(t), "CD2_1": s * np.sin(t), "CD2_2": s * np.cos(t),
            "CUNIT1": "deg", "CUNIT2": "deg"}


@pytest.mark.parametrize("rot,img_angle,pa", [(0, 90, 0), (0, 0, 90), (30, 90, 30), (30, 120, 60), (-45, 45, 90)])
def test_image_angle_to_sky_position_angle(rot, img_angle, pa):
    g = P.sky_geometry(_wcs(rot), 300, 300, 400, [img_angle])
    diff = (g["pa_deg"][0] - pa + 90) % 180 - 90                   # son ejes: 0° = 180°
    assert abs(diff) < 0.5, (rot, img_angle, g["pa_deg"])
    assert abs(g["center_dec_deg"] - 68.16) < 0.05


def test_wcs_info_reads_pixel_scale_from_a_header():
    from astropy.io import fits
    info = P.wcs_info(fits.Header(_wcs(20)))
    assert abs(info["pixel_scale_arcsec"] - 0.11) < 1e-6 and info["wcs"]["CTYPE1"] == "RA---TAN"
    assert P.wcs_info(fits.Header({"NAXIS": 2})) == {}


def test_full_analysis_records_psf_and_sky_geometry():
    """De punta a punta (analysis_job): el JSON lleva la difracción usada, el centro y el PA."""
    import json

    import analysis_job as aj
    nu_c = P.cutoff_cycles_per_px("JWST", "MIRI", "F2100W", 0.11)
    raw = _diffraction_field(2.5, nu_c, 0, n=400, big=1024).astype(np.float32)
    lo, hi = np.percentile(raw, [2, 98])
    crop = np.clip((raw - lo) / (hi - lo), 0, 1).astype(np.float32)
    meta = {"is_fits": True, "format": "FITS", "telescope": "JWST", "instrument": "MIRI", "filter": "F2100W",
            "pixel_scale_arcsec": 0.11, "wcs": _wcs(30), "crop": {"x0": 300, "y0": 300, "side": 400},
            "empty_area_fraction": 0.0}
    out = aj.run_analysis(crop=crop, crop_raw=raw, valid=np.ones(crop.shape, bool),
                          active=["kolmogorov_1941", "anisotropy"], n_null=19, seed=80, name="jw_x_mirimage_i2d.fits",
                          meta_full=meta, source={"archive": "MAST"}, hole_frac=0.0, max_empty_fraction=0.001)
    card = json.loads(out["json"])
    assert abs(card["analysis"]["psf"]["nu_c_cycles_per_px"] - 0.165) < 0.002
    assert card["analysis"]["pixel_scale_arcsec"] == 0.11 and card["analysis"]["descriptors_version"] == 4
    assert card["descriptors"]["kolmogorov_1941"]["psf_corrected"] is True
    assert card["analytics"]["spectrum"]["fit"]["psf_corrected"] is True
    assert abs(card["source"]["center_dec_deg"] - 68.16) < 0.1
    assert 0 <= card["descriptors"]["anisotropy"]["structure_pa_deg"] < 180
