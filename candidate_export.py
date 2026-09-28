"""Tarjeta JSON para spectral-identifier-v1."""
from __future__ import annotations
from datetime import datetime, timezone
import json

SCHEMA_VERSION = "cosmic-candidate.v0.1"
FDR_SCOPE = ["aniso", "flatness", "entropy", "energy_mean"]

def build_candidate(filename, plugin_results, materials, provenance, nos, monte_carlo=None, metadata=None, source_url=""):
    meta = metadata or {}
    mc = monte_carlo or {}
    fdr = dict(mc.get("fdr") or {})
    fdr.setdefault("scope", FDR_SCOPE)
    fdr.setdefault("note", "FDR solo sobre estadisticos baratos del nulo de fase. Los demas plugins van not_tested.")
    nos_pub = dict(nos or {})
    emp = dict(nos_pub.get("empirical") or {})
    emp.pop("nos_combined", None)
    nos_pub["empirical"] = emp
    nos_pub.pop("nos_v1", None)
    return {
        "schema": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "state": materials.get("state") or "known_or_weak",
        "source": {
            "filename": filename, "url": source_url, "format": meta.get("format"),
            "instrument": meta.get("instrument"), "filter": meta.get("filter"),
            "width": meta.get("width"), "height": meta.get("height"),
            "is_fits": bool(meta.get("is_fits")), "provenance": provenance,
        },
        "morphology": {
            "family": materials.get("dominant_family"), "label": materials.get("dominant_label"),
            "family_scores": materials.get("family_scores"),
            "family_margin": materials.get("family_margin"),
            "family_tie": materials.get("family_tie"),
            "tied_families": materials.get("tied_families"),
            "lab_analog": materials.get("lab_analog"),
            "sky_analog": materials.get("sky_analog"),
            "is_candidate": bool(materials.get("is_candidate")),
            "state": materials.get("state"),
            "verdict": materials.get("verdict"),
            "instrument_warning": bool(materials.get("instrument_warning")),
        },
        "nos_morphological": nos_pub,
        "structure_test": {
            "method": mc.get("method"), "p_value": mc.get("p_value"), "z_score": mc.get("z_score"),
            "n_simulations": mc.get("n_simulations"), "fdr": fdr,
            "p_critical": fdr.get("p_critical"), "fdr_pass": fdr.get("fdr_pass"),
        },
        "descriptors": _thin_descriptors(plugin_results),
        "followup": {"action": materials.get("followup"), "needs_spectrum": True, "reject_if": "state in (reject, exploratory_only, known_or_weak)"},
        "spectrum": {"status": "missing", "kind": None, "xy": None, "notes": "Pegar espectro de la MISMA region."},
        "spectral_identifier_handoff": {"target": "WilmerGaspar/spectral-identifier-v1", "ready": False, "reason": "Sin xy espectral no hay identificacion."},
    }

def _thin_descriptors(plugin_results):
    keep = {
        "fractal_base": ("d0", "d1", "d2", "lacunarity", "multifractality_index"),
        "kolmogorov_1941": ("beta", "beta_se", "r_squared", "k_range", "n_k_bins", "k62_kurtosis", "intermittency_factor", "intermittency_clipped", "isotropic_score"),
        "periodicity": ("periodicity_score", "n_significant_peaks", "lattice_hint", "likely_instrument_artifact"),
        "anisotropy": ("anisotropy_index", "dominant_direction_degrees"),
        "persistent_homology": ("betti_0", "betti_1", "euler_characteristic"),
        "renormalization_group": ("correlation_length", "scale_invariance_score"),
        "lyapunov_stability": ("max_lyapunov", "dynamics_type"),
        "entropy": ("shannon_entropy_bits", "normalized_entropy"),
        "fibonacci": ("fibonacci_score", "n_phi_pairs", "n_pairs_tested", "best_ratio", "phi_error"),
        "graph_morphology": ("n_nodes", "n_edges", "n_components", "clustering", "spectral_gap"),
    }
    out = {}
    for plugin, keys in keep.items():
        src = plugin_results.get(plugin)
        if isinstance(src, dict):
            out[plugin] = {k: src.get(k) for k in keys if k in src}
    return out

def dumps_candidate(card):
    return json.dumps(card, indent=2, ensure_ascii=False, default=str)
