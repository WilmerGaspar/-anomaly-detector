"""Mapa de analogos: fisica de materiales <-> morfologia en el cielo."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

FAMILIES = {
    "aggregate": {"label": "Agregado fractal", "lab_analog": "Hollin, dust aggregates, aerogeles, DLA, flocs coloidales", "sky_analog": "Polvo interestelar / nubes con estructura autosimilar", "followup": "Extincion + hielo/silicatos en IR"},
    "cascade": {"label": "Cascada turbulenta", "lab_analog": "Medios intermitentes", "sky_analog": "ISM turbulento proyectado", "followup": "beta en varios filtros; cubo de velocidad"},
    "filament": {"label": "Red filamentaria", "lab_analog": "Polimeros, textura, nanowires", "sky_analog": "Filamentos moleculares", "followup": "Polarimetria"},
    "lattice": {"label": "Orden periodico", "lab_analog": "Cristal / cuasicristal", "sky_analog": "Casi siempre artefacto", "followup": "Descartar fringing"},
    "compact": {"label": "Condensado compacto", "lab_analog": "Grano, nucleacion", "sky_analog": "Estrella o nudo denso", "followup": "Fotometria multi-banda"},
    "critical": {"label": "Casi critico / scale-free", "lab_analog": "Percolacion, opalescencia critica", "sky_analog": "Campos autosimilares", "followup": "Coarse-grain 2x y 4x"},
    "featureless": {"label": "Sin organizacion extra", "lab_analog": "Vidrio / ruido", "sky_analog": "Fondo", "followup": "No es candidato por morfologia"},
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

def interpret(plugin_results, structure_z=0.0, p_value=1.0, fdr_pass=None):
    scores = family_scores(plugin_results, structure_z)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top_key, top_p = ranked[0]
    second_key, second_p = ranked[1]
    artifact = bool(plugin_results.get("periodicity", {}).get("likely_instrument_artifact", False))
    per_score = _f(plugin_results, "periodicity", "periodicity_score")
    phase_null_empty = p_value >= 0.1 and structure_z < 1.5
    n_sig = int(plugin_results.get("periodicity", {}).get("n_significant_peaks") or 0)
    lattice_evidence = (top_key == "lattice" and per_score >= 0.35) or (per_score >= 0.50 and n_sig >= 2)
    if artifact and lattice_evidence:
        top_key = "lattice"; candidate = False
        verdict = "Periodicidad alineada al detector. Artefacto hasta demostrar lo contrario."
    elif lattice_evidence and per_score >= 0.40:
        top_key = "lattice"; candidate = True
        verdict = "Firma periodica en |FFT|. Descartar instrumento, luego espectro."
    elif phase_null_empty:
        top_key = "featureless"; candidate = False
        verdict = "No hay organizacion extra frente a un campo con el mismo espectro."
    elif structure_z >= 2.5:
        candidate = True
        verdict = "Estructura coherente. Familia: %s. Candidato morfologico, no identificacion." % FAMILIES[top_key]["label"]
    else:
        candidate = top_p > 0.28 and structure_z >= 1.5
        verdict = "Senal moderada. Mezcla %s + %s." % (FAMILIES[top_key]["label"], FAMILIES[second_key]["label"])
    if fdr_pass is False:
        candidate = False
        verdict = "FDR no deja ningun descriptor. " + verdict
    elif fdr_pass is True and not candidate and structure_z >= 1.2:
        candidate = True
        verdict = "FDR deja al menos un descriptor. " + verdict
    meta = FAMILIES[top_key]
    return {
        "dominant_family": top_key,
        "dominant_label": meta["label"],
        "dominant_weight": float(top_p),
        "family_scores": scores,
        "lab_analog": meta["lab_analog"],
        "sky_analog": meta["sky_analog"],
        "followup": meta["followup"],
        "instrument_warning": artifact,
        "is_candidate": candidate,
        "verdict": verdict,
        "fdr_pass": fdr_pass,
    }
