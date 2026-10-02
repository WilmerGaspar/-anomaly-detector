"""Alerta de descubrimiento con controles: solo se enciende si la señal sobrevive a
todas las explicaciones conocidas que la app sabe comprobar.

Ninguna alerta afirma un descubrimiento. El nivel mas alto dice: "estructura que el
ruido equivalente no explica y que no se debe a ninguno de los confusores medidos;
requiere revision humana y datos independientes".
"""
from __future__ import annotations

import numpy as np

from scoring import cheap_descriptor_null_details, fdr_decision


# ---------------------------------------------------------------- fuentes puntuales

def _fill_value(lin):
    finite = lin[np.isfinite(lin)]
    return float(np.median(finite)) if finite.size else 0.0


def find_point_sources(raw, nsig=10.0, sigmas=(1.0, 2.0)):
    """Fuentes puntuales en la region a resolucion completa (valores lineales)."""
    from plugins.ridges import compact_sources_linear
    lin = np.asarray(raw, dtype=np.float64)
    lin = np.where(np.isfinite(lin), lin, _fill_value(lin))
    src, thr = compact_sources_linear(lin, sigmas=sigmas, nsig=nsig)
    return src, lin


def _fill_smooth_plus_noise(lin, masked, rng, sigma=4.0):
    """Relleno de las zonas enmascaradas: interpolacion suave desde los pixeles vecinos
    (convolucion normalizada) + ruido gaussiano con la amplitud de la parte de alta
    frecuencia de la imagen. Un relleno con pixeles al azar rompe la correlacion espacial
    del fondo y el nulo lo lee como estructura (medido: con zonas grandes el FDR pasaba
    en ruido puro)."""
    from scipy.ndimage import gaussian_filter
    ok = ~masked
    num = gaussian_filter(np.where(ok, lin, 0.0), sigma)
    den = gaussian_filter(ok.astype(float), sigma)
    smooth = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.median(lin[ok]) if ok.any() else 0.0)
    resid = (lin - smooth)[ok]
    noise = 1.4826 * float(np.median(np.abs(resid - np.median(resid)))) if resid.size else 0.0
    out = lin.copy()
    out[masked] = smooth[masked] + noise * rng.standard_normal(int(masked.sum()))
    return out


MAX_MASK_FRACTION = 0.10
MASK_NSIG = 5.0     # medido: a 10 sigma quedaban estrellas debiles que mantenian el FDR (1 de 4 campos)


def mask_point_sources(raw, seed=0, max_radius=25, nsig=MASK_NSIG, spike_pixels=None, small_shape=None, band=3):
    """Enmascara cada fuente puntual (disco que crece hasta fondo + 3 ruido) y, si se dan,
    los picos de difraccion como bandas de +-`band` px a lo largo de su cresta (los picos son
    lineas largas: un disco que los contuviera tapaba 60-95 % de la imagen)."""
    from scipy.ndimage import binary_dilation
    src, lin = find_point_sources(raw, nsig=nsig)
    rng = np.random.default_rng(seed)
    h, w = lin.shape
    masked = np.zeros(lin.shape, dtype=bool)
    radii = []
    n_extended = 0
    if len(src):
        from plugins.ridges import local_background
        resid = lin - local_background(lin)
        noise = 1.4826 * float(np.median(np.abs(resid - np.median(resid)))) or float(resid.std()) or 1.0
        yy, xx = np.mgrid[:h, :w]
        keep = []
        for y, x in src:
            r = 2
            while r < max_radius:
                y0, y1, x0, x1 = max(0, y - r - 1), y + r + 2, max(0, x - r - 1), x + r + 2
                ring = np.abs(np.hypot(yy[y0:y1, x0:x1] - y, xx[y0:y1, x0:x1] - x) - r) < 0.75
                vals = resid[y0:y1, x0:x1][ring]
                if vals.size and np.median(vals) <= 3 * noise:
                    break
                r += 1
            if r >= max_radius:
                # El perfil no baja al fondo local: es emision extendida, no una fuente puntual.
                n_extended += 1
                continue
            keep.append((y, x))
            r = int(np.ceil(1.5 * r))             # margen para alas de la PSF
            radii.append(r)
            masked |= (yy - y) ** 2 + (xx - x) ** 2 <= r * r
        src = np.array(keep).reshape(-1, 2)
    n_spike_px = 0
    if spike_pixels and small_shape:
        sy, sx = h / small_shape[0], w / small_shape[1]
        sk = np.zeros(lin.shape, dtype=bool)
        for y, x in spike_pixels:
            sk[int(y * sy):int((y + 1) * sy) + 1, int(x * sx):int((x + 1) * sx) + 1] = True
        yy_d, xx_d = np.mgrid[-band:band + 1, -band:band + 1]
        sk = binary_dilation(sk, structure=(yy_d ** 2 + xx_d ** 2 <= band * band))
        n_spike_px = int(sk.sum())
        masked |= sk
    out = _fill_smooth_plus_noise(lin, masked, rng) if masked.any() else lin.copy()
    frac = float(masked.mean())
    return out, {"n_masked": int(len(src)), "n_extended_rejected": int(n_extended), "masked_fraction": frac,
                 "radii": radii, "spike_area_fraction": n_spike_px / masked.size, "nsig": nsig,
                 # Rellenar mas del 10 % de la imagen fabrica estructura (medido: emision extendida
                 # sin estrellas pasaba de FDR 0/4 a 3-4/4 con 17-24 % enmascarado).
                 "valid": frac <= MAX_MASK_FRACTION}


def stretch01(a, lo=2, hi=98):
    finite = a[np.isfinite(a)]
    p_lo, p_hi = np.percentile(finite, [lo, hi])
    out = np.where(np.isfinite(a), a, np.median(finite))
    return np.clip((out - p_lo) / (p_hi - p_lo), 0, 1) if p_hi > p_lo else np.zeros_like(out)


def fdr_without_point_sources(raw, n_simulations=99, seed=81, spike_pixels=None, small_shape=None):
    """Repite el FDR con las fuentes puntuales y sus picos de difraccion enmascarados.
    Si sigue pasando, la estructura no se explica solo por ellas."""
    filled, info = mask_point_sources(raw, seed=seed, spike_pixels=spike_pixels, small_shape=small_shape)
    det = cheap_descriptor_null_details(stretch01(filled), n_simulations=n_simulations, seed=seed)
    fdr = fdr_decision({k: d["p"] for k, d in det.items()}, n_simulations=n_simulations)
    info = {k: v for k, v in info.items() if k != "radii"}
    return {"fdr_pass": bool(fdr["fdr_pass"]), "n_passed": int(fdr["n_passed"]), "p_values": fdr["p_values"],
            "passed_descriptors": fdr["passed_descriptors"], **info}


def fdr_replicate(crop, n_simulations=99, seed=1234):
    """Mismo test con otra semilla: un resultado que depende de la semilla no es robusto."""
    det = cheap_descriptor_null_details(crop, n_simulations=n_simulations, seed=seed)
    fdr = fdr_decision({k: d["p"] for k, d in det.items()}, n_simulations=n_simulations)
    return {"fdr_pass": bool(fdr["fdr_pass"]), "n_passed": int(fdr["n_passed"]), "seed": seed}


# ---------------------------------------------------------------- novedad frente a referencias

NOVELTY_KEYS = (
    ("descriptors", "fractal_base", "d0"), ("descriptors", "fractal_base", "lacunarity"),
    ("descriptors", "kolmogorov_1941", "beta"), ("descriptors", "anisotropy", "anisotropy_index"),
    ("descriptors", "persistent_homology", "betti_0"), ("descriptors", "persistent_homology", "betti_1"),
    ("descriptors", "ridges", "filament_excess"), ("descriptors", "periodicity", "periodicity_score"),
)


def _get(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    try:
        v = float(d)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def card_vector(card):
    """Vector de descriptores de una tarjeta JSON (la que descarga la app)."""
    return {"/".join(p[1:]): _get(card, p) for p in NOVELTY_KEYS}


def novelty_vs_references(card, references, min_refs=5):
    """Rareza frente a analisis previos del usuario (tarjetas JSON): z robusto por descriptor
    (mediana / 1.4826 MAD). Solo dice 'distinto de lo que tu ya analizaste', no 'nuevo para
    la ciencia'."""
    refs = [card_vector(r) for r in references]
    x = card_vector(card)
    out = {"n_references": len(refs), "per_descriptor": {}, "max_abs_z": None, "available": len(refs) >= min_refs}
    if not out["available"]:
        out["note"] = "Hacen falta al menos %d análisis previos (JSON) como referencia." % min_refs
        return out
    zs = []
    for k, v in x.items():
        col = np.array([r[k] for r in refs if r.get(k) is not None], dtype=float)
        if v is None or col.size < min_refs:
            continue
        med = float(np.median(col))
        mad = 1.4826 * float(np.median(np.abs(col - med)))
        scale = mad if mad > 0 else (float(col.std()) or None)
        if not scale:
            continue
        z = (v - med) / scale
        out["per_descriptor"][k] = {"value": v, "ref_median": med, "z": float(z)}
        zs.append(abs(z))
    out["max_abs_z"] = float(max(zs)) if zs else None
    return out


# ---------------------------------------------------------------- semaforo

LEVELS = {
    "invalid": ("🔴", "No válido", "El dato o la región no permiten concluir nada."),
    "none": ("⚪", "Sin estructura", "Nada que el ruido equivalente no explique."),
    "unconfirmed": ("🟠", "Sin confirmar", "Hay señal, pero faltan controles por hacer: no se puede descartar un confusor."),
    "explained": ("🟡", "Explicado por un confusor conocido", "Hay señal, pero la explica algo conocido."),
    "robust": ("🟢", "Estructura robusta", "Sobrevive a todos los controles. Candidata a revisión humana."),
    "pioneer": ("🟣", "Alerta de pionero", "Robusta y atípica frente a la referencia. Requiere datos independientes "
                "y revisión experta antes de afirmar nada."),
}

RIDGE_MIN = 0.006            # medido: ruido/redes/estrellas <= 0.0006; filamentos 0.011-0.030
NOVELTY_Z = 5.0              # |z| robusto frente a tus referencias


def evaluate(card, masked=None, replicate=None, references=None):
    """Semaforo + lista de comprobaciones. `card` es la tarjeta JSON del analisis; `masked`
    y `replicate` vienen de fdr_without_point_sources y fdr_replicate. Una comprobacion que
    no se pudo hacer cuenta como NO superada: nunca se asume a favor de la alerta."""
    morph, st = card.get("morphology", {}), card.get("structure_test", {})
    prov = card.get("source", {}).get("provenance", {})
    an = card.get("analysis", {})
    desc = card.get("descriptors", {})
    rid = desc.get("ridges", {})
    checks = []

    def add(name, ok, detail, blocking=False):
        checks.append({"check": name, "ok": ok, "detail": detail, "blocking": blocking})

    empty = an.get("empty_area_fraction")
    add("Región sin zonas vacías", empty is not None and empty <= 0.001,
        "zona vacía: %s" % ("—" if empty is None else "%.2f %%" % (100 * empty)), blocking=True)
    add("Dato científico (imagen calibrada)", prov.get("verdict") == "usable_science",
        "procedencia: %s" % prov.get("verdict"), blocking=True)
    add("Sin artefacto de instrumento", not morph.get("instrument_warning"),
        morph.get("instrument_warning_reason") or "ninguno detectado", blocking=True)

    fdr = st.get("fdr", {})
    fdr_pass = bool(fdr.get("fdr_pass"))
    ridge = rid.get("filament_excess")
    ridge_pass = ridge is not None and ridge >= RIDGE_MIN
    add("Señal frente al nulo (FDR o crestas)", fdr_pass or ridge_pass,
        "FDR %s/%s; exceso de crestas %s" % (fdr.get("n_passed", 0), fdr.get("n_tested", 0),
                                             "—" if ridge is None else "%.4f (umbral %.3f)" % (ridge, RIDGE_MIN)))

    n_src = morph.get("n_point_sources") or 0
    n_spk = rid.get("n_spike_components") or 0
    if masked is not None and not masked.get("valid", True):
        msg = ("prueba no válida: habría que enmascarar el %.0f %% del área (máximo %.0f %%); no comprobado"
               % (100 * masked["masked_fraction"], 100 * MAX_MASK_FRACTION))
        add("No la explican las fuentes puntuales", False, msg)
        add("No la explican los picos de difracción", n_spk == 0, msg if n_spk else "sin picos de difracción")
    elif masked is not None:
        survives = masked["fdr_pass"] or ridge_pass
        n_m = masked.get("n_masked", 0)
        detail = "%d fuentes (≥ %.0fσ) y sus picos enmascarados (%.1f %% del área) → FDR %d/4%s" % (
            n_m, masked.get("nsig", 5), 100 * masked["masked_fraction"], masked["n_passed"],
            "; crestas %.4f" % ridge if ridge_pass else "")
        add("No la explican las fuentes puntuales", survives if (n_m or n_src) else True,
            detail if (n_m or n_src) else "sin fuentes puntuales")
        add("No la explican los picos de difracción", survives if n_spk else True,
            detail if n_spk else "sin picos de difracción")
    else:
        add("No la explican las fuentes puntuales", False, "prueba de enmascararlas: no comprobado")
        add("No la explican los picos de difracción", n_spk == 0,
            "%d componentes radiales; prueba de enmascararlos: no comprobado" % n_spk if n_spk else "sin picos de difracción")
    if replicate is not None:
        add("Reproducible con otra semilla", replicate["fdr_pass"] or ridge_pass or not fdr_pass,
            "semilla %d → FDR %d/4" % (replicate["seed"], replicate["n_passed"]))
    else:
        add("Reproducible con otra semilla", False, "no comprobado")

    blocking_fail = any(c["blocking"] and not c["ok"] for c in checks)
    signal = next(c["ok"] for c in checks if c["check"].startswith("Señal"))
    confounded = any(not c["ok"] for c in checks if not c["blocking"] and not c["check"].startswith("Señal"))
    novelty = novelty_vs_references(card, references) if references else None
    atypical = bool(novelty and novelty.get("available") and (novelty.get("max_abs_z") or 0) >= NOVELTY_Z)

    if blocking_fail:
        level = "invalid"
    elif not signal:
        level = "none"
    elif confounded:
        failed = [c for c in checks if not c["ok"] and not c["blocking"] and not c["check"].startswith("Señal")]
        level = "unconfirmed" if all("no comprobado" in c["detail"] for c in failed) else "explained"
    elif atypical:
        level = "pioneer"
    else:
        level = "robust"
    icon, title, meaning = LEVELS[level]
    return {"level": level, "icon": icon, "title": title, "meaning": meaning, "checks": checks,
            "novelty": novelty, "novelty_threshold_z": NOVELTY_Z}
