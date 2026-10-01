"""Seleccion automatica de la region sin pixeles vacios (app.largest_finite_square)."""
import ast
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "app.py"


def _load():
    # app.py ejecuta Streamlit al importarse: se extrae solo la funcion.
    tree = ast.parse(SRC.read_text())
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("largest_finite_square", "large_holes")]
    ns = {"np": np, "MAX_ANALYSIS_SIDE": 2048}
    exec(compile(ast.Module(body=fns, type_ignores=[]), str(SRC), "exec"), ns)
    return ns["largest_finite_square"], ns["large_holes"]


def test_square_avoids_rotated_mosaic_border():
    f, _ = _load()
    n = 400
    yy, xx = np.mgrid[:n, :n]
    data = np.ones((n, n))
    data[np.abs(xx - n / 2) + np.abs(yy - n / 2) > n / 2] = np.nan      # mosaico girado 45 grados
    y0, x0, side = f(data)
    assert np.all(np.isfinite(data[y0:y0 + side, x0:x0 + side]))
    assert side >= 195                                                   # optimo ~ n/2


def test_full_and_empty():
    f, _ = _load()
    assert f(np.ones((100, 120)))[2] == 100
    assert f(np.full((50, 50), np.nan)) is None


def test_scattered_bad_pixels_do_not_shrink_region():
    f, holes = _load()
    rng = np.random.default_rng(0)
    data = np.ones((300, 300))
    data[rng.random((300, 300)) < 0.01] = np.nan            # pixeles malos sueltos
    data[:, :60] = np.nan                                   # borde vacio
    h = holes(data)
    assert h[:, :60].all() and h[:, 61:].sum() == 0          # col 60 puede tocar el borde
    y0, x0, side = f(data, holes=h)
    assert side >= 235 and x0 >= 60
