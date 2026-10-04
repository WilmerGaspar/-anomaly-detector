"""Fractal y topologia: valores medidos y repetibles (antes eran np.random.uniform)."""
import numpy as np
import pytest

from plugins.fractal_base import FractalBase
from plugins.persistent_homology import PersistentHomology


def _stretch(a):
    lo, hi = np.percentile(a, [2, 98])
    return np.clip((a - lo) / (hi - lo), 0, 1)


def _noise(n=384, seed=0):
    return _stretch(np.random.default_rng(seed).normal(size=(n, n)))


def _blobs(n=384, k=20, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:n, :n]
    g = sum(np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 10 ** 2)) for cy, cx in rng.uniform(0, n, (k, 2)))
    return _stretch(g + 0.05 * rng.normal(size=(n, n)))


@pytest.mark.parametrize("make", [_noise, _blobs])
def test_repeatable(make):
    img = make()
    assert FractalBase().analyze(img) == FractalBase().analyze(img)
    assert PersistentHomology().analyze(img) == PersistentHomology().analyze(img)


def test_lacunarity_and_multifractality_separate_noise_from_clumps():
    n, b = FractalBase().analyze(_noise()), FractalBase().analyze(_blobs())
    assert n["lacunarity"] < 1.2 and n["multifractality_index"] < 0.05       # repartido, monofractal
    assert b["lacunarity"] > 2.0 and b["multifractality_index"] > 0.05       # grumos con huecos
    for r in (n, b):
        assert 0 <= r["d2"] <= r["d1"] + 0.05 <= 3.05                         # D2 <= D1 (salvo ajuste)


def test_d0_same_as_previous_loop():
    img = _blobs(n=301)                                                       # bordes incompletos

    def old_count(image, box):
        h, w = image.shape
        return sum(bool(np.any(image[i:i + box, j:j + box] > 0.1)) for i in range(0, h, box) for j in range(0, w, box))
    fb = FractalBase()
    for box in (2, 3, 7, 16, 75):
        assert fb._box_count(img, box) == old_count(img, box)


def test_betti_on_known_shapes():
    yy, xx = np.mgrid[:200, :200]
    r = np.hypot(yy - 100, xx - 100)
    ring = ((r > 40) & (r < 55)).astype(float)
    two = np.zeros((200, 200))
    two[20:60, 20:60] = 1
    two[120:180, 100:150] = 1
    ph = PersistentHomology()
    assert (ph.analyze(ring)["betti_0"], ph.analyze(ring)["betti_1"]) == (1, 1)
    assert (ph.analyze(two)["betti_0"], ph.analyze(two)["betti_1"]) == (2, 0)
    assert ph.analyze(np.ones((50, 50)))["betti_0"] == 0                     # sin variacion: vacio
