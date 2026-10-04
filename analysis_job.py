"""Analisis completo de una region (antes dentro de app.py; el codigo es el mismo).

En un modulo aparte para poder probarlo sin Streamlit. Se probo ejecutarlo en un proceso
hijo para devolver la memoria al sistema: el hijo vuelve a cargar numpy, scipy, skimage...
(~300 MB) y el pico total subia. Lo que si funciono esta en app.py (_tune_malloc).
"""
from __future__ import annotations


def run_plugins(image, active, raw=None):
    """`raw`: la region sin estirar; solo la usa `ridges` para separar picos de difraccion."""
    out = {}
    class_steps = [("fractal_base", "plugins.fractal_base", "FractalBase"), ("kolmogorov_1941", "plugins.kolmogorov_1941", "Kolmogorov1941"),
                   ("lyapunov_stability", "plugins.lyapunov_stability", "LyapunovStability"), ("persistent_homology", "plugins.persistent_homology", "PersistentHomology"),
                   ("renormalization_group", "plugins.renormalization_group", "RenormalizationGroup")]
    fn_steps = [("anisotropy", "plugins.anisotropy", "calculate_anisotropy"), ("entropy", "plugins.entropy", "calculate_entropy"),
                ("periodicity", "plugins.periodicity", "analyze_periodicity"), ("fibonacci", "plugins.fibonacci", "analyze_fibonacci"),
                ("graph_morphology", "plugins.graph_morphology", "analyze_graph"),
                ("ridges", "plugins.ridges", "analyze_ridges")]
    for key, mod, cls in class_steps:
        if key in active:
            try:
                m = __import__(mod, fromlist=[cls])
                out[key] = getattr(m, cls)().analyze(image)
            except Exception as exc:
                out[key] = {"error": str(exc)}
    for key, mod, fn in fn_steps:
        if key in active:
            try:
                m = __import__(mod, fromlist=[fn])
                out[key] = getattr(m, fn)(image, raw=raw) if key == "ridges" else getattr(m, fn)(image)
            except Exception as exc:
                out[key] = {"error": str(exc)}
    return out


def run_analysis(crop, crop_raw, valid, active, n_null, seed, name, meta_full, source, hole_frac, max_empty_fraction):
    """Devuelve el diccionario que la app guarda en st.session_state["results"] (sin "key")."""
    from analytics import full_analytics
    from candidate_export import build_candidate, dumps_candidate
    from discovery import evaluate, fdr_replicate, fdr_without_point_sources, find_point_sources
    from golden import golden_window
    from materials_map import interpret
    from nos_morphological import morphological_nos
    from provenance import assess_provenance
    from scoring import cheap_descriptor_null_details, cheap_score_from_image, fdr_decision, surrogate_null_test

    src = source.get("download_url") or source.get("uri") or source.get("archive", "")
    prov = assess_provenance(str(name), meta_full, src)
    results = run_plugins(crop, active, raw=crop_raw)
    mc = surrogate_null_test(crop, None, cheap_score_from_image, n_simulations=n_null, seed=seed)
    details = cheap_descriptor_null_details(crop, n_simulations=n_null, seed=seed + 1)
    fdr = fdr_decision({k: d["p"] for k, d in details.items()}, n_simulations=n_null)
    analytics = full_analytics(crop, seed=seed + 2, valid=valid)
    mc["fdr"] = fdr
    mc["descriptor_nulls"] = {k: {kk: vv for kk, vv in d.items() if kk != "null"} for k, d in details.items()}
    materials = interpret(results, float(mc.get("z_score") or 0), float(mc.get("p_value") or 1),
                          fdr_pass=bool(fdr.get("fdr_pass")), provenance=prov, metadata=meta_full,
                          empty_fraction=hole_frac, max_empty_fraction=max_empty_fraction)
    nos = morphological_nos(results)
    cand = build_candidate(name, results, materials, prov, nos, mc, meta_full, src)
    cand["analytics"] = analytics
    # Controles de la alerta de descubrimiento (discovery.py) y ventana phi (golden.py).
    replicate = fdr_replicate(crop, n_simulations=n_null, seed=seed + 1000)
    rid = results.get("ridges") or {}
    # Siempre: estrellas debiles (por debajo del umbral del aviso, 10 sigma) tambien
    # disparan la curtosis de incrementos; la prueba enmascara desde 5 sigma.
    masked = fdr_without_point_sources(crop_raw, n_simulations=n_null, seed=seed + 2000,
                                       spike_pixels=rid.get("spike_pixels"), small_shape=rid.get("small_shape"))
    golden = golden_window(crop, find_point_sources(crop_raw)[0], seed=seed)
    gate = evaluate(cand, masked=masked, replicate=replicate)
    cand["discovery_gate"] = {k: v for k, v in gate.items() if k != "novelty"}
    cand["discovery_controls"] = {"replicate": replicate, "point_sources_masked": masked}
    cand["golden_window"] = {"alpha": golden["alpha"], "alert": golden["alert"],
                             "tests": [{k: v for k, v in t.items() if k != "null"} for t in golden["tests"]]}
    cand["morphology"]["is_candidate"] = gate["level"] in ("robust", "pioneer")
    # El estado y el veredicto siguen al semaforo: antes podian decir "interes morfologico"
    # mientras el semaforo decia "explicado por un confusor" (HST WFPC2 real).
    new_state = {"none": "known_or_weak", "explained": "explained_by_confounder",
                 "unconfirmed": "unconfirmed"}.get(gate["level"])
    if new_state and materials.get("state") not in ("reject", "exploratory_only", "invalid_region"):
        failed = [c["check"] + ": " + c["detail"] for c in gate["checks"] if not c["ok"]]
        verdict = "%s %s. %s" % (gate["icon"], gate["title"], " · ".join(failed) if failed else gate["meaning"])
        for target in (materials, cand["morphology"]):
            target["state"], target["verdict"] = new_state, verdict
        cand["state"] = new_state
    return {"results": results, "mc": mc, "fdr": fdr, "materials": materials, "nos": nos, "prov": prov,
            "json": dumps_candidate(cand), "details": details, "analytics": analytics, "card": cand,
            "masked": masked, "replicate": replicate, "golden": golden}
