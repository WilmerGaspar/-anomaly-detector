"""Imagen en disco (app.FieldImage): lo calculado por bloques coincide con la imagen en memoria."""
import ast
import gzip
import os
import shutil
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

SRC = Path(__file__).resolve().parents[1] / "app.py"
NAMES = ("_open_fits", "_fits_reader", "_array_reader", "FieldImage", "HoleMap", "largest_clean_square",
         "parse_field", "list_image_hdus")


def _load():
    # app.py ejecuta Streamlit al importarse: se extraen solo estas definiciones.
    tree = ast.parse(SRC.read_text())
    defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in NAMES]
    ns = {"np": np, "os": os, "MAX_ANALYSIS_SIDE": 2048, "PREVIEW_SIDE": 900}
    exec(compile(ast.Module(body=defs, type_ignores=[]), str(SRC), "exec"), ns)
    return ns


def _write(path, hdus):
    fits.HDUList([fits.PrimaryHDU()] + hdus).writeto(path, overwrite=True)
    return str(path)


def _diamond(h, w, scale=0.5):
    yy, xx = np.mgrid[:h, :w]
    d = np.random.default_rng(h).normal(size=(h, w)).astype(np.float32)
    d[np.abs(xx - w / 2) / w + np.abs(yy - h / 2) / h > scale] = np.nan
    return d


@pytest.mark.parametrize("shape,rows", [((400, 300), 512), ((2600, 2500), 512), ((3001, 1999), 7), ((1025, 4100), 100)])
def test_blocks_match_in_memory(shape, rows):
    ns = _load()
    d = _diamond(*shape)
    f = ns["FieldImage"](*ns["_array_reader"](d), rows_per_block=rows)
    hm = ns["HoleMap"](d)
    assert f.holes.f == hm.f and f.holes.shape == hm.shape
    np.testing.assert_array_equal(f.holes.coarse, hm.coarse)
    assert f.holes.largest_square() == hm.largest_square()
    np.testing.assert_array_equal(f.thumb, d[::f.step, ::f.step])
    assert max(f.thumb.shape) <= 900
    fin = np.isfinite(d)
    r, c = np.where(fin.any(1))[0], np.where(fin.any(0))[0]
    assert f.bbox == (r[0], r[-1] + 1, c[0], c[-1] + 1)


def test_fits_scaling_nan_and_crop(tmp_path):
    rng = np.random.default_rng(0)
    sci = rng.normal(5, 2, (1300, 517)).astype(np.float32)      # 1300 filas: bloques de 512 + resto
    sci[:40] = np.nan
    ints = fits.ImageHDU(rng.integers(-3000, 3000, (700, 300)).astype(np.int16), name="RAW")
    ints.header["BSCALE"], ints.header["BZERO"] = 0.5, 100.0
    path = _write(tmp_path / "a.fits", [fits.ImageHDU(sci, name="SCI"), ints])
    ns = _load()
    field, meta = ns["parse_field"](path, "a.fits", 1)
    assert field.shape == (1300, 517) and (meta["height"], meta["width"]) == (1300, 517)
    assert field.bbox == (40, 1300, 0, 517)
    crop = field.crop(100, 17, 500)
    assert crop.dtype == np.float32
    np.testing.assert_array_equal(crop, sci[100:600, 17:517])
    crop[:] = -1                                                  # la copia devuelta no altera la cache
    np.testing.assert_array_equal(field.crop(100, 17, 500), sci[100:600, 17:517])
    np.testing.assert_array_equal(field.crop(0, 0, 50), sci[:50, :50])          # NaN en el mismo sitio
    with fits.open(path) as h:
        ref = h[2].data.astype(np.float32)                        # astropy aplica BSCALE/BZERO
    np.testing.assert_allclose(ns["parse_field"](path, "a.fits", 2)[0].crop(0, 0, 300), ref[:300, :300])


def test_degenerate_axis(tmp_path):
    cube = np.arange(3 * 1025, dtype=np.float32).reshape(1, 3, 1025)    # (1, h, w): 2D tras squeeze
    path = _write(tmp_path / "c.fits", [fits.ImageHDU(cube, name="SCI")])
    ns = _load()
    assert [h["shape"] for h in ns["list_image_hdus"](path)] == [(1, 3, 1025)]
    field, _ = ns["parse_field"](path, "c.fits", 1)
    assert field.shape == (3, 1025)
    np.testing.assert_array_equal(field.crop(0, 1000, 3), cube[0, :, 1000:1003])


def test_gzip_and_tile_compressed(tmp_path):
    img = np.random.default_rng(1).normal(size=(600, 400)).astype(np.float32)
    plain = _write(tmp_path / "g.fits", [fits.ImageHDU(img, name="SCI")])
    with open(plain, "rb") as src, gzip.open(str(tmp_path / "g.fits.gz"), "wb") as dst:
        shutil.copyfileobj(src, dst)
    ns = _load()
    field, _ = ns["parse_field"](str(tmp_path / "g.fits.gz"), "g.fits.gz", 1)
    np.testing.assert_array_equal(field.crop(200, 0, 400), img[200:600])
    ints = np.random.default_rng(2).integers(0, 5000, (500, 450)).astype(np.int32)     # RICE sin perdidas
    comp = _write(tmp_path / "z.fits", [fits.CompImageHDU(ints, name="SCI")])
    field, _ = ns["parse_field"](comp, "z.fits", 1)
    np.testing.assert_array_equal(field.crop(50, 50, 400), ints[50:450, 50:450].astype(np.float32))


def test_all_nan():
    ns = _load()
    assert ns["FieldImage"](*ns["_array_reader"](np.full((50, 60), np.nan, dtype=np.float32))).bbox is None


def test_memory_does_not_hold_the_image(tmp_path):
    # Antes la extension entera quedaba en memoria (y con memmap + copia, 2x: 10000x10000 llegaba
    # a 830 MB en la app). Ahora: miniatura, mapa de vacios, un bloque de 512 filas y la region;
    # medido: ~15 MB de pico tanto con 64 MB como con 128 MB de imagen.
    ns = _load()
    small = _write(tmp_path / "s.fits", [fits.ImageHDU(np.ones((64, 64), dtype=np.float32), name="SCI")])
    ns["parse_field"](small, "s.fits", 1)[0].crop(0, 0, 32)       # calentamiento: importaciones de astropy
    img = np.ones((12000, 1500), dtype=np.float32)               # 72 MB
    path = _write(tmp_path / "m.fits", [fits.ImageHDU(img, name="SCI")])
    del img
    tracemalloc.start()
    field, _ = ns["parse_field"](path, "m.fits", 1)
    field.crop(0, 0, 1024)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 0.3 * 12000 * 1500 * 4, peak / 1e6


def test_cleanup_removes_only_idle_sessions(tmp_path, monkeypatch):
    import time
    tree = ast.parse(SRC.read_text())
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_cleanup_stale_fields"]
    ns = {"os": os, "shutil": shutil, "FIELD_MAX_IDLE_H": 6}
    exec(compile(ast.Module(body=defs, type_ignores=[]), str(SRC), "exec"), ns)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    old = time.time() - 7 * 3600
    for name in ("cms80_idle", "cms80_active", "cms80_mine", "otra_app"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "campo.fits").write_bytes(b"x")
    (tmp_path / "cms80_url_a.fits").write_bytes(b"x")
    for p in ("cms80_idle", "cms80_idle/campo.fits", "cms80_mine", "cms80_mine/campo.fits", "otra_app",
              "otra_app/campo.fits", "cms80_url_a.fits", "cms80_active"):
        os.utime(tmp_path / p, (old, old))                   # cms80_active: su archivo se uso ahora
    ns["_cleanup_stale_fields"](keep=str(tmp_path / "cms80_mine"))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cms80_active", "cms80_mine", "otra_app"]
