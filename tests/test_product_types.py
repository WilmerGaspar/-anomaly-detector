"""Tipos de producto JWST: planos de espectrografo y adquisiciones no son imagenes del cielo."""
from materials_map import interpret
from mast_client import is_image_product
from provenance import assess_provenance

URL = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/"


def test_nirspec_cal_is_not_listed_as_sky_image():
    assert not is_image_product("jw01192008001_02120_00001_nrs1_cal.fits")
    assert not is_image_product("jw01192008001_02120_00001_nrs2_cal.fits")
    assert not is_image_product("jw01234001001_02101_00001_mirifushort_cal.fits")
    assert is_image_product("jw01192006001_0310h_00001_mirimage_i2d.fits")
    assert is_image_product("jw02739001001_02101_00001_nrcb1_cal.fits")


def _prov(name, exp_type):
    meta = {"is_fits": True, "instrument": "NIRSPEC", "exp_type": exp_type}
    return assess_provenance(name, meta, URL + name)


def test_exp_type_spectroscopy_and_acquisition():
    spec = _prov("jw01192008001_02120_00001_nrs1_cal.fits", "NRS_MSASPEC")
    acq = _prov("jw01192008001_02120_00001_nrs1_cal.fits", "NRS_MSATA")
    assert spec["verdict"] == acq["verdict"] == "exploratory_only"
    assert any("NRS_MSASPEC" in r and "espectrografo" in r for r in spec["reasons"])
    assert any("NRS_MSATA" in r and "adquisicion" in r for r in acq["reasons"])


def test_exploratory_verdict_names_the_real_reason():
    prov = _prov("jw01192008001_02120_00001_nrs1_cal.fits", "NRS_MSASPEC")
    v = interpret({}, 3.0, 0.01, fdr_pass=True, provenance=prov)
    assert v["state"] == "exploratory_only" and not v["is_candidate"]
    assert "espectrografo" in v["verdict"] and "uncal" not in v["verdict"]
    det = assess_provenance("jw01_nrcb1_rate.fits", {"is_fits": True}, URL)
    v2 = interpret({}, 3.0, 0.01, fdr_pass=True, provenance=det)
    assert "detector" in v2["verdict"].lower()
