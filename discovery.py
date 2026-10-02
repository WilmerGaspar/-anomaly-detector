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


def mask_point_sources(raw, seed=0, max_radius=25, nsig=10.0):
    """Sustituye cada fuente puntual por pixeles tomados al azar del anillo que la rodea.
    El radio crece hasta que el perfil cae a fondo + 3 ruido (tope `max_radius`).
    Rellenar con un valor constante crearia mesetas que el nulo leeria como estructura."""
    src, lin = find_point_sources(raw, nsig=nsig)
    out = lin.copy()
    if not len(src):
        return out, {"n_masked": 0, "masked_fraction": 0.0, "radii": []}
    rng = np.random.default_rng(seed)
    bg = float(np.median(lin))
    noise = 1.4826 * float(np.median(np.abs(lin - bg))) or float(lin.std()) or 1.0
    h, w = lin.shape
    yy, xx = np.mgrid[:h, :w]
    masked = np.zeros(lin.shape, dtype=bool)
    radii = []
    for y, x in src:
        r = 2
        while r < max_radius:
            ring = (np.abs(np.hypot(yy[max(0, y - r - 1):y + r + 2, max(0, x - r - 1):x + r + 2] - y,
                                    xx[max(0, y - r - 1):y + r + 2, max(0, x - r - 1):x + r + 2] - x) - r) < 0.75)
            vals = lin[max(0, y - r - 1):y + r + 2, max(0, x - r - 1):x + r + 2][ring]
            if vals.size and np.median(vals) <= bg + 3 * noise:
                break
            r += 1
        r = int(np.ceil(1.5 * r))                 # margen para alas de la PSF
        radii.append(r)
        masked |= (yy - y) ** 2 + (xx - x) ** 2 <= r * r
    # Relleno: pixeles al azar de un anillo exterior de cada fuente (fuera de toda mascara).
    for (y, x), r in zip(src, radii):
        disc = ((yy - y) ** 2 + (xx - x) ** 2 <= r * r)
        ann = (((yy - y) ** 2 + (xx - x) ** 2 <= (2 * r) ** 2) & ~masked)
        pool = lin[ann]
        if pool.size < 10:
            pool = lin[~masked]
        if pool.size:
            out[disc] = rng.choice(pool, size=int(disc.sum()), replace=True)
    return out, {"n_masked": int(len(src)), "masked_fraction": float(masked.mean()), "radii": radii}


def stretch01(a, lo=2, hi=98):
    finite = a[np.isfinite(a)]
    p_lo, p_hi = np.percentile(finite, [lo, hi])
    out = np.where(np.isfinite(a), a, np.median(finite))
    return np.clip((out - p_lo) / (p_hi - p_lo), 0, 1) if p_hi > p_lo else np.zeros_like(out)


def fdr_without_point_sources(raw, n_simulations=99, seed=81):
    """Repite el FDR con las fuentes puntuales enmascaradas. Si sigue pasando, la estructura
    no se explica solo por ellas."""
    filled, info = mask_point_sources(raw, seed=seed)
    det = cheap_descriptor_null_details(stretch01(filled), n_simulations=n_simulations, seed=seed)
    fdr = fdr_decision({k: d["p"] for k, d in det.items()}, n_simulations=n_simulations)
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
    if masked is not None and n_src:
        add("No la explican las fuentes puntuales", masked["fdr_pass"] or ridge_pass,
            "%d fuentes enmascaradas (%.1f %% del área) → FDR %d/4" % (masked["n_masked"], 100 * masked["masked_fraction"],
                                                                     masked["n_passed"]))
    elif n_src >= 3:
        add("No la explican las fuentes puntuales", False,
            "%d fuentes puntuales y no se hizo la prueba de enmascararlas: no comprobado" % n_src)
    else:
        add("No la explican las fuentes puntuales", True, "%d fuentes puntuales" % n_src)
    n_spk = rid.get("n_spike_components") or 0
    add("Sin picos de difracción", n_spk == 0, "%d componentes radiales desde fuentes brillantes" % n_spk)
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
