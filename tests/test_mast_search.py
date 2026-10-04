"""Busqueda en MAST: filtros en el servidor y una sola pagina (mast_client.search_observations_detailed).

Antes se descargaban todas las observaciones de todas las misiones en el radio y se filtraba
en la app: con HLSP + HUDF (de lo mas observado del cielo) la app se quedaba sin memoria."""
import pytest

import mast_client


class _FakeObs:
    def __init__(self, n_images=1200, n_mission=3000, n_total=250000):
        self.calls = []
        self.n_images, self.n_mission, self.n_total = n_images, n_mission, n_total

    def query_object(self, *a, **k):
        pytest.fail("query_object descarga todas las misiones: no debe usarse")

    def query_criteria(self, pagesize=None, page=None, **crit):
        self.calls.append(("rows", pagesize, page, crit))
        coll = crit["obs_collection"][0]
        return [{"obs_collection": coll, "dataproduct_type": "image", "obsid": str(i), "obs_id": "o%d" % i,
                 "target_name": "HUDF", "instrument_name": "ACS/WFC"} for i in range(min(pagesize, self.n_images))]

    def query_criteria_count(self, **crit):
        self.calls.append(("count", crit))
        return self.n_images if crit.get("dataproduct_type") == "image" else self.n_mission

    def query_object_count(self, **kw):
        return self.n_total


def test_filters_on_server_and_one_page(monkeypatch):
    fake = _FakeObs()
    monkeypatch.setattr(mast_client, "_obs", lambda: fake)
    out = mast_client.search_observations_detailed("HUDF", ["HLSP"], radius_deg=0.12, limit=500)
    rows_calls = [c for c in fake.calls if c[0] == "rows"]
    assert len(rows_calls) == 1
    _, pagesize, page, crit = rows_calls[0]
    assert (pagesize, page) == (500, 1)                     # una pagina: page=None las bajaria todas
    assert crit["obs_collection"] == ["HLSP"] and crit["dataproduct_type"] == "image"
    assert crit["objectname"] == "HUDF" and crit["radius"] == "0.12 deg"
    assert (out["n_total"], out["n_mission"], out["n_images"], out["n_shown"]) == (250000, 3000, 1200, 500)
    assert out["truncated"] is True


def test_hst_aliases_and_counts_failure(monkeypatch):
    fake = _FakeObs(n_images=3)
    fake.query_object_count = lambda **kw: (_ for _ in ()).throw(RuntimeError("timeout"))
    monkeypatch.setattr(mast_client, "_obs", lambda: fake)
    out = mast_client.search_observations_detailed("M51", ["HST"])
    assert fake.calls[0][3]["obs_collection"] == ["HLA", "HST"]
    assert out["n_total"] is None and out["n_shown"] == 3 and out["truncated"] is False
