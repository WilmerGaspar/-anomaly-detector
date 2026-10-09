"""Datos en periodo de acceso exclusivo (🔒): MAST los lista pero la descarga da 401.

Caso real: «Orion Nebula» en JWST, jw07534130001_03103_00002_mirimage_i2d.fits -> HTTPError 401
Unauthorized y el piloto automático se detenía con "No pude completarlo"."""
import shutil
from pathlib import Path

import pytest

import guide as G
import mast_client as M
from test_autopilot_app import _fits

ROOT = Path(__file__).resolve().parents[1]
TODAY = M._mjd_today()
URL = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/jw07534130001_03103_00002_mirimage_i2d.fits"


@pytest.mark.parametrize("rights,release,public", [
    ("PUBLIC", None, True), ("EXCLUSIVE_ACCESS", TODAY + 200, False), ("EXCLUSIVE_ACCESS", TODAY - 3, True),
    ("EXCLUSIVE_ACCESS", None, False), ("RESTRICTED", None, False), ("", None, True), ("", TODAY + 10, False)])
def test_is_public(rights, release, public):
    assert M.is_public(rights, release) is public


def test_row_carries_access_and_release_date():
    r = M._row({"obs_id": "a", "dataRights": "EXCLUSIVE_ACCESS", "t_obs_release": TODAY + 200})
    assert r["is_public"] is False and r["release_date"][:2] == "20"
    assert M._row({"obs_id": "b", "dataRights": "PUBLIC"})["is_public"] is True


class _FakeObs:
    def __init__(self, result=None, exc=None, table=()):
        self.result, self.exc, self.table = result, exc, list(table)

    def download_file(self, uri, local_path):
        if self.exc:
            raise self.exc
        return self.result

    def query_criteria(self, **kw):
        return self.table

    def query_object_count(self, **kw):
        return len(self.table)

    def query_criteria_count(self, **kw):
        return len(self.table)


@pytest.mark.parametrize("fake", [
    _FakeObs(result=("ERROR", "HTTPError: 401 Client Error: Unauthorized for url: " + URL, URL)),   # lo que se vio
    _FakeObs(exc=RuntimeError("401 Client Error: Unauthorized for url: " + URL)),
])
def test_401_becomes_a_clear_exclusive_access_error(monkeypatch, fake):
    monkeypatch.setattr(M, "_obs", lambda: fake)
    with pytest.raises(M.ExclusiveAccessError, match="acceso exclusivo"):
        M.download_product("mast:x", "jw07534130001_03103_00002_mirimage_i2d.fits", size_mb=10)


def test_other_failures_stay_ordinary_errors(monkeypatch):
    monkeypatch.setattr(M, "_obs", lambda: _FakeObs(result=("ERROR", "HTTPError: 503 Server Error", URL)))
    with pytest.raises(RuntimeError) as e:
        M.download_product("mast:x", "f.fits", size_mb=10)
    assert not isinstance(e.value, M.ExclusiveAccessError)


def test_search_lists_public_first_and_counts_locked(monkeypatch):
    table = [{"obs_id": "lock", "obs_collection": "JWST", "dataproduct_type": "image", "dataRights": "EXCLUSIVE_ACCESS",
              "t_obs_release": TODAY + 100},
             {"obs_id": "pub", "obs_collection": "JWST", "dataproduct_type": "image", "dataRights": "PUBLIC"}]
    monkeypatch.setattr(M, "_obs", lambda: _FakeObs(table=table))
    res = M.search_observations_detailed("Orion Nebula", ["JWST"])
    assert [r["obs_id"] for r in res["rows"]] == ["pub", "lock"] and res["n_locked"] == 1


def test_guide_never_picks_locked_rows_or_files():
    rows = [{"obs_id": "a", "is_public": False}, {"obs_id": "b", "is_public": True}, {"obs_id": "c"}]
    assert [r["obs_id"] for r in G.rows_to_try(rows)] == ["b", "c"]
    prods = [{"filename": "x_i2d.fits", "is_image": True, "too_big": False, "uri": "u1", "is_public": False},
             {"filename": "y_i2d.fits", "is_image": True, "too_big": False, "uri": "u2", "is_public": True}]
    assert G.pick_product(prods)["filename"] == "y_i2d.fits"


@pytest.fixture
def orion(tmp_path, monkeypatch):
    src = tmp_path / "src_i2d.fits"
    _fits(src, 0)
    calls = {"list": [], "download": []}
    base = {"filters": "F770W", "obs_collection": "JWST", "instrument_name": "MIRI", "target_name": "ORION",
            "dataproduct_type": "image"}
    rows = [M._row(dict(base, obs_id="obs_bloqueada", obsid="1", dataRights="EXCLUSIVE_ACCESS", t_obs_release=TODAY + 90)),
            M._row(dict(base, obs_id="obs_401", obsid="2", dataRights="PUBLIC")),      # MAST no la marca, pero da 401
            M._row(dict(base, obs_id="obs_buena", obsid="3", dataRights="PUBLIC"))]

    def search(target, missions, radius_deg=0.12, limit=500, images_only=True):
        return {"target": target, "radius_deg": radius_deg, "missions": list(missions), "n_total": 438, "n_mission": 438,
                "n_images": 438, "n_shown": 3, "truncated": False, "rows": rows, "n_locked": 1}

    def list_products(obsid, max_mb=None):
        calls["list"].append(obsid)
        return [{"filename": "jw_%s_mirimage_i2d.fits" % obsid, "is_image": True, "too_big": False, "is_public": True,
                 "uri": "mast:%s" % obsid, "size_mb": 1, "download_url": "https://mast/%s" % obsid, "hint": ""}]

    def download(uri, filename, max_mb=400, size_mb=None):
        calls["download"].append(uri)
        if uri == "mast:2":
            raise M.ExclusiveAccessError(M.EXCLUSIVE_MSG % filename)
        out = tmp_path / ("dl_%s" % filename)
        shutil.copy(src, out)
        return str(out)

    monkeypatch.setattr(M, "search_observations_detailed", search)
    monkeypatch.setattr(M, "list_fits_products", list_products)
    monkeypatch.setattr(M, "download_product", download)
    return calls


def test_autopilot_skips_locked_data_and_keeps_going(orion):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    at.run()
    next(b for b in at.main.button if b.key == "guide_auto").click().run()
    for _ in range(4):
        if "guide_log" in at.session_state and at.session_state["guide_log"]:
            break
        at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert orion["list"] == ["2", "3"]                       # la 🔒 ni se lista
    assert orion["download"] == ["mast:2", "mast:3"]         # tras el 401 sigue con la siguiente
    notes = " ".join(at.session_state["guide_note"])
    assert "Salté 1 observación 🔒" in notes and "MAST respondió 401" in notes and "No pude completarlo" not in notes
    assert at.session_state["guide_log"][0]["file"] == "jw_3_mirimage_i2d.fits"
