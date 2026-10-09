"""Piloto automático de la guía, de punta a punta en la app real (streamlit.testing), con MAST
simulado: buscar -> elegir observación -> elegir archivo -> descargar -> analizar -> guía + bitácora."""
import shutil
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import mast_client

ROOT = Path(__file__).resolve().parents[1]


def _fits(path, seed):
    rng = np.random.default_rng(seed)
    n = 300
    k = np.hypot(np.fft.fftfreq(n)[:, None], np.fft.fftfreq(n)[None])
    k[0, 0] = 1
    img = np.fft.ifft2(np.fft.fft2(rng.normal(size=(n, n))) * k ** -1.4).real.astype(np.float32)
    hdr = fits.Header({"EXTNAME": "SCI", "FILTER": "F200W", "INSTRUME": "NIRCAM", "TARGNAME": "PRUEBA"})
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(img, header=hdr)]).writeto(path, overwrite=True)


@pytest.fixture
def fake_mast(tmp_path, monkeypatch):
    src = tmp_path / "src_i2d.fits"
    _fits(src, 0)
    calls = {"search": 0, "list": [], "download": []}
    # Filas construidas con la misma función que usa la app con MAST real.
    rows = [mast_client._row({"obs_id": "obs_sin_imagen", "obsid": "1", "filters": "F200W", "obs_collection": "JWST",
                              "instrument_name": "NIRCAM", "target_name": "M16", "dataproduct_type": "image"}),
            mast_client._row({"obs_id": "obs_buena", "obsid": "2", "filters": "F200W", "obs_collection": "JWST",
                              "instrument_name": "NIRCAM", "target_name": "M16", "dataproduct_type": "image"})]

    def search(target, missions, radius_deg=0.12, limit=500, images_only=True):
        calls["search"] += 1
        return {"target": target, "radius_deg": radius_deg, "missions": list(missions), "n_total": 10, "n_mission": 2,
                "n_images": 2, "n_shown": 2, "truncated": False, "rows": rows}

    def list_products(obsid, max_mb=None):
        calls["list"].append(obsid)
        if obsid == "1":                                   # la primera observacion no tiene imagen util
            return [{"filename": "x_uncal.fits", "is_image": False, "too_big": False, "uri": "mast:u", "size_mb": 5,
                     "download_url": "", "hint": "crudo"}]
        return [{"filename": "jw_prueba_i2d.fits", "is_image": True, "too_big": False, "uri": "mast:ok", "size_mb": 1,
                 "download_url": "https://mast/ok", "hint": ""}]

    def download(uri, filename, max_mb=400, size_mb=None):
        calls["download"].append(uri)
        out = tmp_path / ("dl_%d_%s" % (len(calls["download"]), filename))
        shutil.copy(src, out)
        return str(out)

    monkeypatch.setattr(mast_client, "search_observations_detailed", search)
    monkeypatch.setattr(mast_client, "list_fits_products", list_products)
    monkeypatch.setattr(mast_client, "download_product", download)
    return calls


def test_one_click_search_load_analyze(fake_mast):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    at.run()
    assert not at.exception
    guide_btn = [b for b in at.main.button if b.key == "guide_auto"]
    assert guide_btn, "la guía debe ofrecer ▶ Hazlo por mí al empezar"
    guide_btn[0].click().run()
    for _ in range(4):                                    # st.rerun encadena: buscar/cargar -> analizar -> guía
        if at.session_state["guide_log"] if "guide_log" in at.session_state else None:
            break
        at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert fake_mast["search"] == 1 and fake_mast["list"] == ["1", "2"] and fake_mast["download"] == ["mast:ok"]
    log = at.session_state["guide_log"]
    assert len(log) == 1 and log[0]["file"] == "jw_prueba_i2d.fits"
    notes = " ".join(at.session_state["guide_note"])
    assert "obs_sin_imagen" in notes and "jw_prueba_i2d.fits" in notes
    main_text = " ".join(m.value for m in at.main.markdown)
    assert "último análisis" in main_text                   # la guía ya interpreta el resultado
    assert not [b for b in at.sidebar.button if b.key.startswith("guide_")]   # en pantalla principal


def test_other_filter_reports_when_nothing_left(fake_mast):
    """Tras analizar, ▶ Otro filtro prueba las observaciones no usadas; si ninguna sirve, lo dice."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    at.run()
    next(b for b in at.main.button if b.key == "guide_auto").click().run()
    for _ in range(4):
        if "guide_log" in at.session_state and at.session_state["guide_log"]:
            break
        at.run()
    other = [b for b in at.main.button if b.key == "guide_other_filter"]
    assert other, "con un resultado de MAST la guía ofrece ▶ Otro filtro"
    other[0].click().run()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    notes = " ".join(at.session_state["guide_note"])
    assert "Ninguna de las 1 observaciones" in notes          # solo quedaba la que no tiene imagen
    assert fake_mast["search"] == 1                              # reutiliza la búsqueda, no repite la consulta
