"""Descargas a disco (mast_client.download_product / download_url): devuelven una ruta, no bytes."""
import functools
import http.server
import os
import threading
from pathlib import Path

import pytest

import mast_client


class _FakeObservations:
    def __init__(self, payload, status="COMPLETE"):
        self.payload, self.status = payload, status

    def download_file(self, uri, local_path):
        Path(local_path).write_bytes(self.payload)
        return (self.status, None, None)


def test_download_product_returns_path_in_own_folder(monkeypatch):
    monkeypatch.setattr(mast_client, "_obs", lambda: _FakeObservations(b"SIMPLE" * 100))
    a = mast_client.download_product("mast:x/a.fits", "a.fits")
    b = mast_client.download_product("mast:x/a.fits", "a.fits")
    try:
        assert Path(a).read_bytes() == b"SIMPLE" * 100
        assert Path(a).parent.name.startswith("cms80_mast_")
        assert Path(a).parent != Path(b).parent              # dos descargas no se pisan
    finally:
        for p in (a, b):
            os.remove(p)
            os.rmdir(Path(p).parent)


@pytest.mark.parametrize("payload,status", [(b"x" * 3 * 1048576, "COMPLETE"), (b"x", "ERROR")])
def test_download_product_failure_leaves_nothing(monkeypatch, tmp_path, payload, status):
    monkeypatch.setattr(mast_client, "_obs", lambda: _FakeObservations(payload, status))
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    with pytest.raises(RuntimeError):
        mast_client.download_product("mast:x/big.fits", "big.fits", max_mb=2)
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def server(tmp_path):
    (tmp_path / "ok.fits").write_bytes(b"A" * 2_500_000)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield "http://127.0.0.1:%d/" % httpd.server_address[1]
    httpd.shutdown()


def test_download_url_streams_to_disk(server, tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr("tempfile.tempdir", str(out))
    path = mast_client.download_url(server + "ok.fits")
    assert os.path.getsize(path) == 2_500_000 and Path(path).parent == out
    os.remove(path)
    with pytest.raises(RuntimeError):                       # Content-Length > tope
        mast_client.download_url(server + "ok.fits", max_mb=1)
    assert list(out.iterdir()) == []                        # sin restos del intento fallido
