"""Seleccion automatica de la region sin zonas vacias (app.HoleMap)."""
import ast
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "app.py"


def _load():
    # app.py ejecuta Streamlit al importarse: se extraen solo estas definiciones.
    tree = ast.parse(SRC.read_text())
    defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in ("HoleMap", "largest_clean_square")]
    ns = {"np": np, "MAX_ANALYSIS_SIDE": 2048}
    exec(compile(ast.Module(body=defs, type_ignores=[]), str(SRC), "exec"), ns)
    return ns["HoleMap"]


def _diamond(h, w, scale=0.5):
    yy, xx = np.mgrid[:h, :w]
    d = np.ones((h, w), dtype=np.float32)
    d[np.abs(xx - w / 2) / w + np.abs(yy - h / 2) / h > scale] = np.nan      # mosaico girado 45 grados
    return d


@pytest.mark.parametrize("shape", [(400, 400), (1019, 1029), (2600, 2500)])
def test_square_has_no_empty_pixels(shape):
    data = _diamond(*shape)
    y0, x0, side = _load()(data).largest_square()
    assert np.all(np.isfinite(data[y0:y0 + side, x0:x0 + side]))
    assert side >= 0.45 * min(shape)                                      # optimo ~ 0.5 * lado


def test_full_and_empty():
    hm = _load()
    assert hm(np.ones((100, 120))).largest_square()[2] == 100
    assert hm(np.full((50, 50), np.nan)).largest_square() is None


def test_scattered_bad_pixels_are_not_holes():
    hm = _load()
    rng = np.random.default_rng(0)
    data = np.ones((300, 300))
    data[rng.random((300, 300)) < 0.01] = np.nan                         # pixeles malos sueltos
    data[:, :60] = np.nan                                                # borde vacio
    m = hm(data)
    assert m.fraction(0, 0, 60) > 0.99
    assert m.fraction(0, 100, 200) == 0.0
    y0, x0, side = m.largest_square()
    assert side >= 235 and x0 >= 60


def test_memory_on_large_mosaic():
    # Antes: ~530 MB de pico con 4236 x 4214 (Streamlit Cloud tiene ~1 GB).
    data = _diamond(4236, 4214, 0.48)
    tracemalloc.start()
    m = _load()(data)
    r = m.largest_square()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 60e6
    assert np.all(np.isfinite(data[r[0]:r[0] + r[2], r[1]:r[1] + r[2]]))
