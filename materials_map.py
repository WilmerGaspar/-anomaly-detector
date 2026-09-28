"""Mapa de analogos + estado honesto del candidato."""
from __future__ import annotations

FAMILIES = {
    "aggregate": {"label": "Agregado fractal", "lab_analog": "Hollin, dust aggregates, aerogeles", "sky_analog": "Polvo / nubes autosimilares", "followup": "Extincion + hielo/silicatos en IR"},
    "cascade": {"label": "Cascada turbulenta", "lab_analog": "Medios intermitentes", "sky_analog": "ISM turbulento proyectado", "followup": "beta en varios filtros"},
    "filament": {"label": "Red filamentaria", "lab_analog": "Polimeros, textura", "sky_analog": "Filamentos moleculares", "followup": "Polarimetria"},
    "lattice": {"label": "Orden periodico", "lab_analog": "Cristal / cuasicristal", "sky_analog": "Casi siempre artefacto", "followup": "Descartar fringing"},
    "compact": {"label": "Condensado compacto", "lab_analog": "Grano, nucleacion", "sky_analog": "Estrella o nudo denso", "followup": "Fotometria multi-banda"},
    "critical": {"label": "Casi critico / scale-free", "lab_analog": "Percolacion", "sky_analog": "Campos autosimilares", "followup": "Coarse-grain 2x y 4x"},
    "featureless": {"label": "Sin organizacion extra", "lab_analog": "Vidrio / ruido", "sky_analog": "Fondo", "followup": "No es candidato por morfologia"},
    "mixed": {"label": "Mezcla (empate)", "lab_analog": "No asignar un solo analogo", "sky_analog": "Campo con varias familias a la vez", "followup": "No titular una familia"},
}

STATE_TEXT = {
    "reject": "Rechazado para descubrimiento: procedencia insuficiente.",
    "exploratory_only": "Producto de detector (uncal/rate). Exploratorio, no ciencia usable.",
    "known_or_weak": "Campo usable pero FDR no deja descriptores. Estructura tipica o test barato insuficiente. No es candidato.",
    "morph_interesting": "FDR deja al menos un descriptor barato. Interes morfologico. Sin espectro no hay identificacion.",
    "needs_spectrum": "Morfologia sobrevive FDR. Siguiente paso: x1d de la MISMA region.",
}

def _f(d, *keys, default=0.0):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    try:
        v = float(cur)
        return default if v != v else v
    except (TypeError, ValueError):
        return default

def _clip(x):
    return float(max(0.0, min(1.0, x)))

def _band(x, lo, hi):
    if hi <= lo:
        return 0.0
    if lo <= x <= hi:
        return 1.0
    width = hi - lo
    return max(0.0, 1.0 - (lo - x) / width) if x < lo else max(0.0, 1.0 - (x - hi) / width)

def family_scores(plugin_results, structure_z=0.0):
    d0 = _f(plugin_results, "fractal_base", "d0", default=1.5)
    multi = _f(plugin_results, "fractal_base", "multifractality_index")
    lac = _f(plugin_results, "fractal_base", "lacunarity", default=1.0)
    beta = _f(plugin_results, "kolmogorov_1941", "beta")
    inter = _f(plugin_results, "kolmogorov_1941", "intermittency_factor")
    iso = _f(plugin_results, "kolmogorov_1941", "isotropic_score", default=1.0)
    aniso = _f(plugin_results, "anisotropy", "anisotropy_index")
    b0 = _f(plugin_results, "persistent_homology", "betti_0")
    b1 = _f(plugin_results, "persistent_homology", "betti_1")
    crit = _f(plugin_results, "renormalization_group", "criticality_score")
    scale = _f(plugin_results, "renormalization_group", "scale_invariance_score")
    per = _f(plugin_results, "periodicity", "periodicity_score")
    artifact = bool(plugin_results.get("periodicity", {}).get("likely_instrument_artifact", False))
    entropy_n = _f(plugin_results, "entropy", "normalized_entropy", default=0.5)
    zpos = min(max(0.0, structure_z) / 6.0, 1.0)
    scores = {
        "aggregate": _clip(0.35 * _band(d0, 1.35, 1.85) + 0.25 * _band(lac, 1.2, 4.0) + 0.20 * _band(multi, 0.15, 1.2) + 0.20 * zpos),
        "cascade": _clip(0.40 * _band(beta, 1.4, 3.2) + 0.30 * inter + 0.15 * iso + 0.15 * zpos),
        "filament": _clip(0.45 * aniso + 0.20 * (1.0 - iso) + 0.20 * _band(b1, 1, 20) + 0.15 * zpos),
        "lattice": _clip((0.70 * per + 0.20 * _band(b1, 2, 30) + 0.10 * (1.0 - entropy_n)) * (0.2 if artifact else 1.0)),
        "compact": _clip(0.35 * _band(d0, 0.8, 1.35) + 0.25 * (1.0 if b0 <= 2 else 0.0) + 0.20 * (1.0 - min(entropy_n, 1.0)) + 0.20 * _band(beta, 3.0, 6.0)),
        "critical": _clip(0.5 * crit + 0.35 * scale + 0.15 * _band(multi, 0.2, 1.5)),
        "featureless": _clip(0.5 * (1.0 - zpos) + 0.3 * (1.0 if structure_z < 1.0 else 0.0) + 0.2 * (1.0 if per < 0.1 and aniso < 0.2 else 0.0)),
    }
    total = sum(scores.values()) or 1.0
    return {k: v / total for k, v in scores.items()}

def decide_state(provenance=None, fdr_pass=None, product_level=None, family_key="featureless"):
    prov = provenance or {}
    verdict = prov.get("verdict")
    level = product_level or prov.get("product_level")
    if verdict in ("reject_for_discovery", "reject") or prov.get("is_fits") is False:
        return "reject"
    if level == "detector" or verdict == "exploratory_only":
        return "exploratory_only"
    if fdr_pass is False:
        return "known_or_weak"
    if fdr_pass is True and family_key != "featureless":
        return "needs_spectrum"
    if fdr_pass is True:
        return "morph_interesting"
    return "known_or_weak"

def interpret(plugin_results, structure_z=0.0, p_value=1.0, fdr_pass=None, provenance=None):
    scores = family_scores(plugin_results, structure_z)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top_key, top_p = ranked[0]
    second_key, second_p = ranked[1]
    margin = float(top_p - second_p)
    mixed = margin < 0.05
    if mixed:
        top_key = "mixed"
    artifact = bool(plugin_results.get("periodicity", {}).get("likely_instrument_artifact", False))
    per_score = _f(plugin_results, "periodicity", "periodicity_score")
    n_sig = int(plugin_results.get("periodicity", {}).get("n_significant_peaks") or 0)
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
    return {
        "dominant_family": top_key,
        "dominant_label": meta["label"],
        "dominant_weight": float(top_p if not mixed else ranked[0][1]),
        "family_scores": scores,
        "family_margin": margin,
        "family_tie": mixed,
        "tied_families": [ranked[0][0], second_key, ranked[2][0]] if mixed else [top_key],
        "lab_analog": meta["lab_analog"],
        "sky_analog": meta["sky_analog"],
        "followup": meta["followup"],
        "instrument_warning": artifact,
        "is_candidate": is_candidate,
        "state": state,
        "verdict": verdict,
        "fdr_pass": fdr_pass,
    }
