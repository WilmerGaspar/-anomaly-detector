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


@pytest.mark.parametrize("shape", [(512, 512), (630, 630), (601, 601), (400, 700)])
def test_uniform_image_has_dimension_two_at_any_size(shape):
    """Solo cajas completas: antes las cajas a medias del borde sesgaban D0 (1.92 en 630 px) y
    el orden salia invertido (D0 < D1 < D2) con la multifractalidad recortada a 0."""
    r = FractalBase().analyze(np.ones(shape))
    assert abs(r["d0"] - 2) < 1e-6 and abs(r["d1"] - 2) < 1e-6 and abs(r["d2"] - 2) < 1e-6
    assert r["multifractality_index"] < 1e-6


def _cascade(levels=10, m=(0.4, 0.3, 0.2, 0.1), seed=1):
    """Cascada multiplicativa 2D (modelo p) con dimensiones exactas conocidas."""
    rng = np.random.default_rng(seed)
    a = np.ones((1, 1))
    for _ in range(levels):
        w = np.array([rng.permutation(m).reshape(2, 2) for _ in range(a.size)]).reshape(a.shape + (2, 2))
        a = (a[:, :, None, None] * w).transpose(0, 2, 1, 3).reshape(a.shape[0] * 2, a.shape[1] * 2)
    return a


@pytest.mark.parametrize("side", [1024, 630, 777])
def test_generalized_dimensions_match_theory_on_a_cascade(side):
    m = np.array([0.4, 0.3, 0.2, 0.1])
    d1, d2 = -np.sum(m * np.log2(m)), -np.log2(np.sum(m ** 2))          # 1.846 y 1.737
    img = _cascade()[:side, :side]
    r = FractalBase().analyze(img / img.max())
    assert abs(r["d1"] - d1) < 0.03 and abs(r["d2"] - d2) < 0.04
    assert abs(r["multifractality_index"] - (2 - d2)) < 0.05             # antes 0.194 con 630 px (teoría 0.263)


def test_entropy_uses_the_real_range():
    """La app pasa la región en [0, 1]: antes el histograma iba de 0 a 255 y daba ~0.14 bits siempre."""
    from plugins.entropy import calculate_entropy
    rng = np.random.default_rng(0)
    assert calculate_entropy(rng.uniform(size=(300, 300)))["normalized_entropy"] > 0.99
    half = np.r_[np.zeros(5000), np.ones(5000)].reshape(100, 100)
    assert abs(calculate_entropy(half)["shannon_entropy_bits"] - 1.0) < 1e-9
    g = np.clip(rng.normal(0.5, 0.15, (300, 300)), 0, 1)
    assert calculate_entropy(g)["shannon_entropy_bits"] > 6
    assert calculate_entropy(g)["shannon_entropy_bits"] == pytest.approx(calculate_entropy(255 * g)["shannon_entropy_bits"])
    assert calculate_entropy(np.full((50, 50), 0.3))["shannon_entropy_bits"] == 0.0


@pytest.mark.parametrize("angle", [0, 30, 60, 90, 135])
def test_anisotropy_direction_follows_the_gradient(angle):
    """Antes se ordenaban los autovalores pero no los autovectores: 60° salía -30° y 90° salía 0°."""
    from plugins.anisotropy import calculate_anisotropy
    n = 400
    yy, xx = np.mgrid[:n, :n]
    t = np.radians(angle)
    img = 0.5 + 0.5 * np.sin(2 * np.pi * (xx * np.cos(t) + yy * np.sin(t)) / 23.0)
    img = img + 0.05 * np.random.default_rng(0).normal(size=(n, n))
    r = calculate_anisotropy(img)
    diff = (r["dominant_direction_degrees"] - angle + 90) % 180 - 90       # ejes: 135° = -45°
    assert abs(diff) < 1.0 and r["anisotropy_index"] > 0.7


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


def _field(n, beta, seed):
    k = np.hypot(np.fft.fftfreq(n)[:, None], np.fft.fftfreq(n)[None])
    k[0, 0] = 1
    return np.fft.ifft2(np.fft.fft2(np.random.default_rng(seed).normal(size=(n, n))) * k ** (-beta / 2)).real


@pytest.mark.parametrize("beta", [2.5, 3.67])
def test_kolmogorov_beta_unbiased_with_a_real_error(beta):
    """Anillos centrados (antes β ~3 % bajo) y beta_se = error de la pendiente (antes era el valor p)."""
    from plugins.kolmogorov_1941 import Kolmogorov1941
    rs = [Kolmogorov1941().analyze(_field(630, beta, s)) for s in range(4)]
    assert abs(np.mean([r["beta"] for r in rs]) - beta) < 0.06
    assert all(0.005 < r["beta_se"] < 0.2 for r in rs)


@pytest.mark.parametrize("beta", [2.5, 3.67])
def test_spectrum_beta_not_steepened_by_the_256px_reduction(beta):
    """Regiones grandes: antes β salía ~0.15 alto (3.87 para 3.67) por el filtro de la reducción."""
    from analytics import full_analytics
    fits = [full_analytics(_field(630, beta, s), seed=s)["spectrum"]["fit"] for s in range(3)]
    assert abs(np.mean([f["beta"] for f in fits]) - beta) < 0.08
    assert all(f["k_min"] == 0.02 and f["k_max"] == 0.25 and np.isfinite(f["intercept"]) for f in fits)


def _crop(beta, seed, n=630, big=1024, stretch=1.0):
    rng = np.random.default_rng(seed)
    ky, kx = np.fft.fftfreq(big)[:, None], np.fft.fftfreq(big)[None]
    k = np.hypot(ky * stretch, kx)
    k[0, 0] = 1
    f = np.fft.ifft2(np.fft.fft2(rng.normal(size=(big, big))) * k ** (-beta / 2)).real
    return f[100:100 + n, 200:200 + n] / f.std()


def test_kolmogorov_beta_on_a_non_periodic_crop_with_a_gradient():
    """Sin ventana, los bordes arrastraban β hacia 3 (3.05 para 3.67; 2.45 frente a 1.66 en MIRI real)."""
    from plugins.kolmogorov_1941 import Kolmogorov1941
    xx = np.mgrid[:630, :630][1]
    for beta in (1.5, 3.67):
        vals = [Kolmogorov1941().analyze(_crop(beta, s) + 3.0 * xx / 630)["beta"] for s in range(2)]
        assert abs(np.mean(vals) - beta) < 0.2, (beta, vals)


def test_orientation_test_finds_alignment_and_ignores_gradients():
    """Dirección preferente frente a campos isótropos con el mismo espectro (fila del campo magnético)."""
    from plugins.anisotropy import orientation_test
    xx = np.mgrid[:630, :630][1]
    iso = orientation_test(_crop(2.5, 0) + 20.0 * xx / 630, n_null=49)        # gradiente fuerte, sin dirección propia
    assert iso["orientation_p"] > 0.05
    aligned = orientation_test(_crop(2.5, 0, stretch=1.5), n_null=49)
    assert aligned["orientation_p"] <= 0.02 and aligned["orientation_coherence"] > aligned["orientation_null_q95"]
