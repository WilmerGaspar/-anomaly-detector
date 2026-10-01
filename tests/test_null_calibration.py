"""Calibracion del nulo: si estos tests fallan, el sistema inventa candidatos.

Ejecutar:  python -m pytest tests/test_null_calibration.py -q
"""
import numpy as np

from scoring import (benjamini_hochberg, cheap_descriptor_pvalues, fdr_decision, iaaft_surrogate,
                     mc_p_value, phase_randomized_surrogate)


def gaussian_field(n, beta, rng):
    ky = np.fft.fftfreq(n)[:, None]
    kx = np.fft.rfftfreq(n)[None, :]
    k = np.sqrt(kx ** 2 + ky ** 2)
    k[0, 0] = 1.0
    f = (rng.standard_normal(k.shape) + 1j * rng.standard_normal(k.shape)) * k ** (-beta / 2)
    f[0, 0] = 0.0
    return np.fft.irfft2(f, s=(n, n))


def filament_field(n, rng, n_lines=12):
    """Control positivo: filamentos finos y nitidos sobre fondo gaussiano."""
    img = 0.3 * gaussian_field(n, 2.5, rng)
    img = (img - img.mean()) / img.std()
    yy, xx = np.mgrid[0:n, 0:n]
    for _ in range(n_lines):
        a = rng.uniform(0, np.pi)
        c = rng.uniform(0, n)
        d = np.abs((xx - n / 2) * np.sin(a) - (yy - n / 2) * np.cos(a) - (c - n / 2))
        img += 3.0 * np.exp(-0.5 * (d / 0.8) ** 2)
    return img


def test_phase_surrogate_preserves_spectrum_exactly():
    rng = np.random.default_rng(0)
    img = gaussian_field(96, 2.8, rng)
    s = phase_randomized_surrogate(img, rng)
    assert np.allclose(np.abs(np.fft.rfft2(s)), np.abs(np.fft.rfft2(img)), rtol=1e-8, atol=1e-8)


def test_iaaft_preserves_histogram_exactly():
    rng = np.random.default_rng(1)
    img = np.exp(gaussian_field(96, 2.8, rng))
    s = iaaft_surrogate(img, rng)
    assert np.array_equal(np.sort(s.ravel()), np.sort(img.ravel()))


def test_p_value_formula():
    assert mc_p_value([0.1] * 99, 1.0) == 0.01
    assert abs(mc_p_value([2.0] + [0.1] * 23, 1.0) - 2 / 25) < 1e-12


def test_bh_known_case():
    rej, _ = benjamini_hochberg([0.01, 0.02, 0.03, 0.5], alpha=0.05)
    assert rej.tolist() == [True, True, True, False]


def _fpr(make_field, n_fields, n_null, seed):
    rng = np.random.default_rng(seed)
    hits = 0
    for i in range(n_fields):
        p = cheap_descriptor_pvalues(make_field(rng), n_simulations=n_null, seed=seed + i)
        hits += fdr_decision(p)["fdr_pass"]
    return hits / n_fields


def test_false_positive_rate_gaussian_fields():
    fpr = _fpr(lambda r: gaussian_field(96, r.uniform(2.0, 3.5), r), n_fields=40, n_null=99, seed=10)
    assert fpr <= 0.15, fpr


def test_false_positive_rate_nonlinear_transform():
    # exp(campo gaussiano) es exactamente la hipotesis nula de IAAFT: no debe salir candidato.
    fpr = _fpr(lambda r: np.exp(gaussian_field(96, 2.8, r)), n_fields=40, n_null=99, seed=500)
    assert fpr <= 0.15, fpr


def test_detects_filaments():
    rng = np.random.default_rng(7)
    hits = sum(fdr_decision(cheap_descriptor_pvalues(filament_field(96, rng), n_simulations=99, seed=i))["fdr_pass"]
               for i in range(10))
    assert hits >= 8, hits
