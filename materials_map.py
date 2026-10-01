"""Mapa de analogos + estado honesto del candidato.

Cambio principal (ver CHANGES.md): un descriptor ausente (plugin apagado o con
error) ya no se imputa con un valor por defecto. Antes, d0 ausente valia 1.5,
que cae dentro de la banda de 'aggregate' y le regalaba puntos. Ahora cada
familia promedia solo los terminos disponibles, renormalizando los pesos.
"""
from __future__ import annotations

FAMILIES = {
    "aggregate": {"label": "Agregado fractal", "lab_analog": "Hollín, agregados de polvo, aerogeles", "sky_analog": "Polvo / nubes autosimilares", "followup": "Extinción + hielos/silicatos en IR"},
    "cascade": {"label": "Cascada turbulenta", "lab_analog": "Medios intermitentes", "sky_analog": "ISM turbulento proyectado", "followup": "β en varios filtros"},
    "filament": {"label": "Red filamentaria", "lab_analog": "Polímeros, texturas", "sky_analog": "Filamentos moleculares", "followup": "Polarimetría"},
    "lattice": {"label": "Orden periódico", "lab_analog": "Cristal / cuasicristal", "sky_analog": "Casi siempre artefacto", "followup": "Descartar fringing"},
    "compact": {"label": "Condensado compacto", "lab_analog": "Grano, nucleación", "sky_analog": "Estrella o nudo denso", "followup": "Fotometría multibanda"},
    "critical": {"label": "Casi crítico / sin escala", "lab_analog": "Percolación", "sky_analog": "Campos autosimilares", "followup": "Coarse-graining 2x y 4x"},
    "featureless": {"label": "Sin organización extra", "lab_analog": "Vidrio / ruido", "sky_analog": "Fondo", "followup": "No es candidato por morfología"},
    "mixed": {"label": "Mezcla (empate)", "lab_analog": "No asignar un solo análogo", "sky_analog": "Campo con varias familias a la vez", "followup": "No titular una familia"},
}

STATE_TEXT = {
    "reject": "Rechazado para descubrimiento: procedencia insuficiente.",
    "exploratory_only": "Producto de detector (uncal/rate). Exploratorio, no ciencia usable.",
    "known_or_weak": "Campo usable, pero ningún descriptor sobrevive al FDR. Estructura típica o test insuficiente. No es candidato.",
    "morph_interesting": "Al menos un descriptor sobrevive al FDR. Interés morfológico. Sin espectro no hay identificación.",
    "needs_spectrum": "La morfología sobrevive al FDR. Siguiente paso: espectro (x1d) de la MISMA región.",
}


def _f(d, *keys, default=0.0):
    v = _g(d, *keys)
    return default if v is None else v


def _g(d, *keys):
    """Valor numerico finito o None si falta / no es numero."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    try:
        v = float(cur)
    except (TypeError, ValueError):
        return None
    return None if v != v or v in (float("inf"), float("-inf")) else v


def _clip(x):
    return float(max(0.0, min(1.0, x)))


def _band(x, lo, hi):
    if x is None:
        return None
    if hi <= lo:
        return 0.0
    if lo <= x <= hi:
        return 1.0
    width = hi - lo
    return max(0.0, 1.0 - (lo - x) / width) if x < lo else max(0.0, 1.0 - (x - hi) / width)


def _map(x, fn):
    return None if x is None else fn(x)


def _max(*vals):
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def _wavg(terms, shared=()):
    """Promedio ponderado ignorando terminos None; renormaliza pesos.
    `shared` son terminos comunes a todas las familias (p. ej. el z del nulo):
    cuentan en el promedio pero NO en la cobertura, para que una familia sin
    ningun descriptor propio no gane solo por z."""
    own = [(w, v) for w, v in terms if v is not None]
    own_total = sum(w for w, _ in terms)
    coverage = sum(w for w, _ in own) / own_total if own_total > 0 else 0.0
    avail = own + [(w, v) for w, v in shared if v is not None]
    wsum = sum(w for w, _ in avail)
    if not own or wsum <= 0:
        return 0.0, 0.0
    return _clip(sum(w * v for w, v in avail) / wsum), coverage


def instrument_warning_reason(plugin_results, metadata=None):
    per = plugin_results.get("periodicity") or {}
    reasons = []
    if per.get("likely_instrument_artifact"):
        reasons.append("picos FFT alineados al detector (n=%s, axis_frac=%s)" % (per.get("n_significant_peaks"), per.get("axis_aligned_fraction")))
    filt = str((metadata or {}).get("filter") or "")
    if "CLEAR" in filt.upper():
        reasons.append("filtro CLEAR — no es banda fotométrica estándar")
    return "; ".join(reasons) if reasons else None


def family_scores(plugin_results, structure_z=0.0, return_coverage=False):
    d0 = _g(plugin_results, "fractal_base", "d0")
    multi = _g(plugin_results, "fractal_base", "multifractality_index")
    lac = _g(plugin_results, "fractal_base", "lacunarity")
    beta = _g(plugin_results, "kolmogorov_1941", "beta")
    inter = _g(plugin_results, "kolmogorov_1941", "intermittency_factor")
    iso = _g(plugin_results, "kolmogorov_1941", "isotropic_score")
    aniso = _g(plugin_results, "anisotropy", "anisotropy_index")
    b0 = _g(plugin_results, "persistent_homology", "betti_0")
    b1 = _g(plugin_results, "persistent_homology", "betti_1")
    crit = _g(plugin_results, "renormalization_group", "criticality_score")
    scale = _g(plugin_results, "renormalization_group", "scale_invariance_score")
    per = _g(plugin_results, "periodicity", "periodicity_score")
    artifact = bool((plugin_results.get("periodicity") or {}).get("likely_instrument_artifact", False))
    entropy_n = _g(plugin_results, "entropy", "normalized_entropy")
    # Crestas largas frente a subrogados IAAFT (plugins/ridges.py). Medido en controles
    # sinteticos: ruido/red/manchas <= 0.001; filamentos rectos o curvos 0.011-0.030.
    ridge = _map(_g(plugin_results, "ridges", "filament_excess"), lambda v: _clip(v / 0.01))
    n_streaks = _g(plugin_results, "periodicity", "n_streaks")
    streak = _map(n_streaks, lambda v: 1.0 if v >= 1 else 0.0)
    z = float(structure_z) if structure_z is not None else 0.0
    zpos = min(max(0.0, z) / 6.0, 1.0)

    raw = {
        "aggregate": _wavg([(0.35, _band(d0, 1.35, 1.85)), (0.25, _band(lac, 1.2, 4.0)), (0.20, _band(multi, 0.15, 1.2))], shared=[(0.20, zpos)]),
        "cascade": _wavg([(0.40, _band(beta, 1.4, 3.2)), (0.30, inter), (0.15, iso)], shared=[(0.15, zpos)]),
        "filament": _wavg([(0.55, _max(ridge, streak)), (0.15, aniso), (0.05, _map(iso, lambda v: 1.0 - v)), (0.10, _band(b1, 1, 20))],
                          shared=[(0.15, zpos)]),
        "lattice": _wavg([(0.70, per), (0.20, _band(b1, 2, 30)), (0.10, _map(entropy_n, lambda v: 1.0 - v))]),
        "compact": _wavg([(0.35, _band(d0, 0.8, 1.35)), (0.25, _map(b0, lambda v: 1.0 if v <= 2 else 0.0)),
                          (0.20, _map(entropy_n, lambda v: 1.0 - min(v, 1.0))), (0.20, _band(beta, 3.0, 6.0))]),
        "critical": _wavg([(0.5, crit), (0.35, scale), (0.15, _band(multi, 0.2, 1.5))]),
        "featureless": _wavg([(0.5, 1.0 - zpos), (0.3, 1.0 if z < 1.0 else 0.0),
                              (0.2, None if per is None or aniso is None else (1.0 if per < 0.1 and aniso < 0.2 else 0.0))]),
    }
    if artifact:
        raw["lattice"] = (raw["lattice"][0] * 0.2, raw["lattice"][1])
    # La evidencia se pondera por cobertura: un descriptor ausente no cuenta a
    # favor ni se inventa; una familia con 1 de 3 descriptores pesa menos.
    scores = {k: v * cov for k, (v, cov) in raw.items()}
    coverage = {k: round(cov, 3) for k, (_, cov) in raw.items()}
    total = sum(scores.values()) or 1.0
    norm = {k: v / total for k, v in scores.items()}
    return (norm, coverage) if return_coverage else norm


def decide_state(provenance=None, fdr_pass=None, product_level=None, family_key="featureless"):
    prov = provenance or {}
    verdict = prov.get("verdict")
    level = product_level or prov.get("product_level")
    if verdict in ("reject_for_discovery", "reject") or prov.get("is_fits") is False:
        return "reject"
    if level == "detector" or verdict == "exploratory_only":
        return "exploratory_only"
    if fdr_pass is True and family_key != "featureless":
        return "needs_spectrum"
    if fdr_pass is True:
        return "morph_interesting"
    return "known_or_weak"


def interpret(plugin_results, structure_z=0.0, p_value=1.0, fdr_pass=None, provenance=None, metadata=None):
    scores, coverage = family_scores(plugin_results, structure_z, return_coverage=True)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top_key, top_p = ranked[0]
    second_key, second_p = ranked[1]
    margin = float(top_p - second_p)
    mixed = margin < 0.05
    if mixed:
        top_key = "mixed"
    per_res = plugin_results.get("periodicity") or {}
    artifact = bool(per_res.get("likely_instrument_artifact", False))
    per_score = _f(plugin_results, "periodicity", "periodicity_score")
    n_sig = int(per_res.get("n_significant_peaks") or 0)
    lattice_evidence = (not mixed and top_key == "lattice" and per_score >= 0.35) or (per_score >= 0.50 and n_sig >= 2)
    if artifact and lattice_evidence:
        top_key = "lattice"
    state = decide_state(provenance, fdr_pass, family_key=top_key if top_key != "mixed" else ranked[0][0])
    is_candidate = state in ("needs_spectrum", "morph_interesting")
    if artifact and lattice_evidence:
        is_candidate = False
        state = "known_or_weak"
    meta = FAMILIES.get(top_key, FAMILIES["mixed"])
    verdict = STATE_TEXT[state]
    if mixed:
        verdict = "Empate de familias (%s vs %s, margen=%.3f). " % (ranked[0][0], second_key, margin) + verdict
    warn_reason = instrument_warning_reason(plugin_results, metadata)
    return {
        "dominant_family": top_key,
        "dominant_label": meta["label"],
        "dominant_weight": float(top_p if not mixed else ranked[0][1]),
        "family_scores": scores,
        "family_coverage": coverage,
        "family_margin": margin,
        "family_tie": mixed,
        "tied_families": [ranked[0][0], second_key, ranked[2][0]] if mixed else [top_key],
        "lab_analog": meta["lab_analog"],
        "sky_analog": meta["sky_analog"],
        "followup": meta["followup"],
        "instrument_warning": artifact or bool(warn_reason),
        "instrument_warning_reason": warn_reason,
        "is_candidate": is_candidate,
        "state": state,
        "verdict": verdict,
        "fdr_pass": fdr_pass,
        "p_value": p_value,
    }
