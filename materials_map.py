"""Mapa de analogos: fisica de materiales <-> morfologia en el cielo."""
from __future__ import annotations

from typing import Dict, List, Tuple

FAMILIES = {
    "aggregate": {
        "label": "Agregado fractal",
        "lab_analog": "Hollin, dust aggregates, aerogeles, DLA, flocs coloidales",
        "sky_analog": "Polvo interestelar / nubes con estructura autosimilar",
        "followup": "Extincion + hielo/silicatos en IR; si hay muestra, Raman/FTIR",
    },
    "cascade": {
        "label": "Cascada turbulenta",
        "lab_analog": "Medios intermitentes, aleaciones en solidificacion turbulenta",
        "sky_analog": "ISM turbulento, viento estelar, turbulencia 2D proyectada",
        "followup": "Espectro de velocidades (si hay cubo); beta en varios filtros",
    },
    "filament": {
        "label": "Red filamentaria",
        "lab_analog": "Polimeros, liquid crystals, materials con textura, nanowires",
        "sky_analog": "Filamentos moleculares, estructuras magnetizadas",
        "followup": "Polarimetria; comparar direccion con campo B publicado",
    },
    "lattice": {
        "label": "Orden periodico",
        "lab_analog": "Cristal, cuasicristal, photonic crystal, superred",
        "sky_analog": "Muy raro en el cielo. Priorizar artefacto de instrumento",
        "followup": "Descartar fringing CCD / spikes. Si sobrevive: espectro de la region",
    },
    "compact": {
        "label": "Condensado compacto",
        "lab_analog": "Grano, nucleacion, precipitado, inclusion",
        "sky_analog": "Estrella, nudo denso, protoplanetary clump",
        "followup": "Fotometria multi-banda; no basta la morfologia",
    },
    "critical": {
        "label": "Casi critico / scale-free",
        "lab_analog": "Percolacion, critical opalescence, transiciones de fase",
        "sky_analog": "Campos autosimilares en un rango de escalas",
        "followup": "Repetir coarse-graining en recortes a 2x y 4x de zoom",
    },
    "featureless": {
        "label": "Sin organizacion extra",
        "lab_analog": "Vidrio / ruido / campo homogeneo",
        "sky_analog": "Fondo, sky-noise, nebulosidad suave",
        "followup": "No es candidato a material nuevo por morfologia",
    },
}


def _f(d: dict, *keys, default=0.0) -> float:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    try:
        v = float(cur)
        if v != v:
            return default
        return v
    except (TypeError, ValueError):
        return default


def family_scores(plugin_results: Dict, structure_z: float = 0.0) -> Dict[str, float]:
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


def interpret(plugin_results: Dict, structure_z: float = 0.0, p_value: float = 1.0) -> Dict:
    scores = family_scores(plugin_results, structure_z)
    ranked: List[Tuple[str, float]] = sorted(scores.items(), key=lambda kv: -kv[1])
    top_key, top_p = ranked[0]
    second_key, second_p = ranked[1]
    artifact = bool(plugin_results.get("periodicity", {}).get("likely_instrument_artifact", False))
    lattice_hint = plugin_results.get("periodicity", {}).get("lattice_hint", "ninguno")
    per_score = _f(plugin_results, "periodicity", "periodicity_score")
    phase_null_empty = p_value >= 0.1 and structure_z < 1.5
    n_sig = int(plugin_results.get("periodicity", {}).get("n_significant_peaks") or 0)
    lattice_evidence = (top_key == "lattice" and per_score >= 0.35) or (per_score >= 0.50 and n_sig >= 2)
    if artifact and lattice_evidence:
        top_key = "lattice"
        top_p = max(top_p, scores.get("lattice", 0.0), per_score)
        verdict = "Hay periodicidad alineada al detector. Tratar como artefacto hasta demostrar lo contrario."
        candidate = False
    elif lattice_evidence and per_score >= 0.40:
        top_key = "lattice"
        top_p = max(top_p, scores.get("lattice", 0.0), per_score)
        verdict = "Firma periodica en |FFT|. Descartar instrumento, luego espectro."
        candidate = True
    elif phase_null_empty:
        verdict = "No hay organizacion extra frente a un campo con el mismo espectro."
        candidate = False
        top_key = "featureless"
        top_p = scores.get("featureless", 1.0)
    elif structure_z >= 2.5:
        verdict = f"Estructura coherente. Familia dominante: {FAMILIES[top_key]['label']}. Candidato morfologico — no es identificacion."
        candidate = True
    else:
        verdict = f"Senal moderada. Mezcla {FAMILIES[top_key]['label']} + {FAMILIES[second_key]['label']}."
        candidate = top_p > 0.28 and structure_z >= 1.5
    meta = FAMILIES[top_key]
    hypotheses = [{
        "family": k,
        "label": FAMILIES[k]["label"],
        "weight": round(float(p), 3),
        "lab_analog": FAMILIES[k]["lab_analog"],
        "sky_analog": FAMILIES[k]["sky_analog"],
    } for k, p in ranked[:3]]
    return {
        "dominant_family": top_key,
        "dominant_label": meta["label"],
        "dominant_weight": float(top_p),
        "family_scores": scores,
        "hypotheses": hypotheses,
        "lab_analog": meta["lab_analog"],
        "sky_analog": meta["sky_analog"],
        "followup": meta["followup"],
        "lattice_hint": lattice_hint,
        "instrument_warning": artifact,
        "is_candidate": candidate,
        "verdict": verdict,
        "materials_interest": float(min(1.0, max(0.0, structure_z) / 8.0 + scores.get("lattice", 0) * 0.5 + scores.get("aggregate", 0) * 0.25 + scores.get("critical", 0) * 0.15)),
    }


def _clip(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _band(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    if lo <= x <= hi:
        return 1.0
    width = hi - lo
    if x < lo:
        return max(0.0, 1.0 - (lo - x) / width)
    return max(0.0, 1.0 - (x - hi) / width)
