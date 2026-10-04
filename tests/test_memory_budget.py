"""Presupuesto de memoria del analisis en la region mas grande que permite la app.

Streamlit Cloud garantiza ~690 MB y la app ya ocupa ~450 MB estabilizada: el analisis no
puede pasar de ~250 MB por encima de su base. Con 2048 px pasaba de 550 MB (la app caia en
el campo profundo de Hubble); por eso MAX_ANALYSIS_SIDE = 1024."""
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = r'''
import numpy as np, analysis_job as aj
import plugins.fractal_base, plugins.kolmogorov_1941, scoring, analytics, discovery, golden
def kb(key): return int([l for l in open("/proc/self/status") if l.startswith(key)][0].split()[1])
side = %d
rng = np.random.default_rng(0)
k = np.hypot(np.fft.fftfreq(side)[:, None], np.fft.fftfreq(side)[None]); k[0, 0] = 1
raw = np.fft.ifft2(np.fft.fft2(rng.normal(size=(side, side))) * k ** -1.3).real.astype(np.float32)
yy, xx = np.mgrid[:side, :side]
for y, x in rng.uniform(10, side - 10, (60, 2)):
    sl = np.s_[int(y) - 8:int(y) + 9, int(x) - 8:int(x) + 9]
    raw[sl] += 30 * np.exp(-((yy[sl] - y) ** 2 + (xx[sl] - x) ** 2) / 3.0)
del yy, xx
lo, hi = np.percentile(raw, [2, 98]); crop = np.clip((raw - lo) / (hi - lo), 0, 1).astype(np.float32)
base = kb("VmRSS")
aj.run_analysis(crop=crop, crop_raw=raw, valid=np.ones(crop.shape, bool),
    active=["fractal_base", "kolmogorov_1941", "periodicity", "anisotropy", "ridges", "persistent_homology",
            "renormalization_group", "entropy", "graph_morphology"],
    n_null=99, seed=80, name="x.fits", meta_full={"is_fits": True, "format": "FITS"}, source={"archive": "t"},
    hole_frac=0.0, max_empty_fraction=0.001)
print((kb("VmHWM") - base) // 1024)
'''


def _max_side():
    tree = ast.parse((ROOT / "app.py").read_text())
    for n in tree.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "MAX_ANALYSIS_SIDE":
            return n.value.value
    raise AssertionError("MAX_ANALYSIS_SIDE no encontrado")


def test_analysis_at_max_side_fits_budget():
    side = _max_side()
    out = subprocess.run([sys.executable, "-c", SCRIPT % side], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    extra_mb = int(out.stdout.strip().splitlines()[-1])
    assert extra_mb < 250, "el analisis con %d px usa %d MB sobre su base" % (side, extra_mb)
