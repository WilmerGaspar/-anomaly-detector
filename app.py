import streamlit as st

st.set_page_config(page_title="CMS-80", page_icon="\u25a0", layout="wide")
st.markdown("""<style>
.stApp { background:#020803; color:#b7ffc2; }
.stButton>button { background:#031108 !important; color:#33ff66 !important; border:1px solid #33ff66 !important; border-radius:0 !important; }
[data-testid="stMetricValue"] { color:#33ff66 !important; }
[data-testid="stSidebar"] { background:#010604 !important; }
</style>""", unsafe_allow_html=True)
st.markdown("## CMS-80  COSMIC MATERIALS SCOUT")
st.caption("i2d/x1d  \u00b7  FDR  \u00b7  Kolmogorov  \u00b7  IsolationForest  \u00b7  grafos  \u00b7  catalogo")

st.sidebar.markdown("`CONFIG`")
source_url = st.sidebar.text_input("URL FITS DIRECTO", value="")
mode = st.sidebar.selectbox("MODO", ["balanced", "explorer", "conservative"])
plugins = {
    "fractal_base": st.sidebar.checkbox("FRACTAL", True),
    "kolmogorov_1941": st.sidebar.checkbox("P(k)", True),
    "periodicity": st.sidebar.checkbox("FFT", True),
    "anisotropy": st.sidebar.checkbox("ANISO", True),
    "persistent_homology": st.sidebar.checkbox("TOPO", True),
    "renormalization_group": st.sidebar.checkbox("RG", True),
    "lyapunov_stability": st.sidebar.checkbox("ROSENSTEIN", True),
    "entropy": st.sidebar.checkbox("SHANNON", True),
    "fibonacci": st.sidebar.checkbox("FIBONACCI / PHI", True),
    "graph_morphology": st.sidebar.checkbox("GRAFOS", True),
}
active = [k for k, v in plugins.items() if v]
n_null = st.sidebar.slider("NULOS", 10, 80, 24, 2)
use_iso = st.sidebar.checkbox("ISOFOREST", True)

CURATED = {
    "JWST": [("NGC 7023", "nube polvo / control MIRI"), ("M16", "pilares"), ("Stephan's Quintet", "grupo"), ("SMACS 0723", "lente"), ("Jupiter", "planeta"), ("PDS 70", "disco")],
    "HST": [("M16", "pilares"), ("M51", "espiral"), ("Crab Nebula", "filamentos"), ("Hoag Object", "anillo raro"), ("Red Square Nebula", "simetria"), ("NGC 7023", "polvo")],
    "ROMAN": [("LMC", "campo amplio"), ("SMC", "campo amplio"), ("Andromeda", "M31")],
    "HLSP": [("HUDF", "profundo"), ("GOODS-S", "survey"), ("COSMOS", "survey")],
}

def run_plugins(image, active):
    out = {}
    class_steps = [
        ("fractal_base", "plugins.fractal_base", "FractalBase"),
        ("kolmogorov_1941", "plugins.kolmogorov_1941", "Kolmogorov1941"),
        ("lyapunov_stability", "plugins.lyapunov_stability", "LyapunovStability"),
        ("persistent_homology", "plugins.persistent_homology", "PersistentHomology"),
        ("renormalization_group", "plugins.renormalization_group", "RenormalizationGroup"),
    ]
    fn_steps = [
        ("anisotropy", "plugins.anisotropy", "calculate_anisotropy"),
        ("entropy", "plugins.entropy", "calculate_entropy"),
        ("periodicity", "plugins.periodicity", "analyze_periodicity"),
        ("fibonacci", "plugins.fibonacci", "analyze_fibonacci"),
        ("graph_morphology", "plugins.graph_morphology", "analyze_graph"),
    ]
    for key, mod, cls in class_steps:
        if key not in active:
            continue
        try:
            m = __import__(mod, fromlist=[cls])
            out[key] = getattr(m, cls)().analyze(image)
        except Exception as exc:
            out[key] = {"error": str(exc)}
    for key, mod, fn in fn_steps:
        if key not in active:
            continue
        try:
            m = __import__(mod, fromlist=[fn])
            out[key] = getattr(m, fn)(image)
        except Exception as exc:
            out[key] = {"error": str(exc)}
    return out

tab_scout, tab_cat, tab_rub = st.tabs(["SCOUT", "CATALOGO MISION", "RUBRICA"])

with tab_scout:
    st.markdown("### DATOS")
    d1, d2, d3 = st.columns([2, 2, 1])
    with d1:
        target = st.text_input("Objeto", value="NGC 7023")
    with d2:
        missions = st.multiselect("Misiones", ["HST", "JWST", "ROMAN", "HLSP"], default=["HST", "JWST"])
    with d3:
        go = st.button("BUSCAR EN MAST", use_container_width=True)
    if go and target.strip():
        try:
            from mast_client import search_observations
            with st.spinner("MAST (45s, 2 intentos)..."):
                st.session_state["mast_obs"] = search_observations(target, missions)
            st.session_state.pop("mast_prods", None)
            if not st.session_state.get("mast_obs"):
                st.warning("Sin filas. Usa CATALOGO o Upload.")
        except Exception as exc:
            st.error(str(exc))
            st.info("MAST a veces no responde. Carga un i2d local.")
    obs_rows = st.session_state.get("mast_obs") or []
    if obs_rows:
        st.dataframe(
            [{"mision": r.get("mission"), "instrumento": r.get("instrument"), "filtro": r.get("filters"), "objeto": r.get("target"), "obs_id": r.get("obs_id")} for r in obs_rows],
            hide_index=True, use_container_width=True,
        )
        labels = ["%s | %s | %s | %s" % (r.get("mission"), r.get("instrument"), r.get("filters"), r.get("obs_id")) for r in obs_rows]
        pick = st.selectbox("Observacion", labels)
        chosen = obs_rows[labels.index(pick)]
        if chosen.get("jpeg_url"):
            st.image(chosen["jpeg_url"], caption="preview archivo")
        if st.button("LISTAR TODOS LOS FITS", use_container_width=True):
            try:
                from mast_client import list_fits_products
                with st.spinner("listando FITS..."):
                    st.session_state["mast_prods"] = list_fits_products(chosen["obsid"])
            except Exception as exc:
                st.error(str(exc))
    prods = st.session_state.get("mast_prods") or []
    if prods:
        filtro = st.radio("Filtro de producto", ["recomendados (i2d/drz/x1d)", "todos", "crudos"], horizontal=True)
        shown = prods
        if filtro.startswith("recomendados"):
            shown = [p for p in prods if any(k in p["filename"].lower() for k in ("i2d", "drz", "drc", "x1d", "s3d"))] or prods
        elif filtro.startswith("crudos"):
            shown = [p for p in prods if any(k in p["filename"].lower() for k in ("uncal", "rate"))] or prods
        st.dataframe([{"archivo": p["filename"], "MB": p.get("size_mb"), "nota": p.get("hint")} for p in shown], hide_index=True, use_container_width=True)
        plabels = ["%s | %s" % (p["filename"], p.get("hint") or "") for p in shown]
        prod = shown[st.selectbox("Selecciona FITS", range(len(plabels)), format_func=lambda i: plabels[i])]
        if st.button("CARGAR FITS ELEGIDO", use_container_width=True):
            try:
                from mast_client import download_product
                with st.spinner("bajando..."):
                    st.session_state["field_bytes"] = download_product(prod["uri"], prod["filename"])
                st.session_state["field_name"] = prod["filename"]
                st.session_state["field_url"] = prod.get("uri") or ""
                st.session_state["started"] = False
            except Exception as exc:
                st.error(str(exc))
    up = st.file_uploader("O carga local", type=["jpg", "jpeg", "png", "tif", "tiff", "fits", "fit"])
    if up is not None:
        st.session_state["field_name"] = up.name
        st.session_state["field_bytes"] = up.getvalue()
        st.session_state["started"] = False
    name = st.session_state.get("field_name")
    raw = st.session_state.get("field_bytes")
    src = st.session_state.get("field_url") or source_url
    if not raw:
        st.info("Carga un i2d/x1d (MAST o Upload) y pulsa INICIAR.")
    else:
        st.success("BUFFER %s" % name)
        if st.button("INICIAR", type="primary", use_container_width=True):
            st.session_state["started"] = True
        if st.session_state.get("started"):
            try:
                import io
                import numpy as np
                from PIL import Image
                from astropy.io import fits
                buf = io.BytesIO(raw)
                low = str(name).lower()
                if low.endswith((".fits", ".fit", ".fits.gz")):
                    with fits.open(buf) as hdul:
                        data = None
                        for hdu in hdul:
                            if getattr(hdu, "data", None) is not None:
                                data = hdu.data
                                if str(getattr(hdu, "name", "")) == "SCI":
                                    break
                        if data is None:
                            st.error("NO SCI DATA"); st.stop()
                        image = np.nan_to_num(np.array(data, dtype=np.float32), nan=0.0)
                        if image.ndim > 2:
                            image = np.squeeze(image)
                            if image.ndim > 2:
                                image = image[0]
                        p2, p98 = np.percentile(image, [2, 98])
                        image = np.clip((image - p2) / (p98 - p2), 0, 1) if p98 > p2 else np.clip(image, 0, 1)
                        header = hdul[0].header
                        meta = {"filename": name, "format": "FITS", "width": int(image.shape[1]), "height": int(image.shape[0]), "is_fits": True, "instrument": str(header.get("INSTRUME", header.get("TELESCOP", ""))), "filter": str(header.get("FILTER", header.get("FILTER1", "")))}
                else:
                    im = Image.open(buf).convert("L")
                    image = np.array(im, dtype=np.float32) / 255.0
                    meta = {"filename": name, "format": im.format, "width": im.width, "height": im.height, "is_fits": False, "instrument": "", "filter": ""}
                from provenance import assess_provenance
                prov = assess_provenance(meta["filename"], meta, src)
                a, b, c, d = st.columns(4)
                a.metric("FILE", str(name)[:22])
                b.metric("SIZE", "%sx%s" % (meta["width"], meta["height"]))
                c.metric("TRUST", "%.2f" % prov["trust_score"])
                d.metric("GATE", prov["verdict"])
                if prov.get("product_level") == "detector":
                    st.warning("Producto de detector (uncal/rate). Exploratorio, no ciencia usable.")
                st.image(image, caption=str(name), use_container_width=True, clamp=True)
            except Exception as exc:
                st.exception(exc); st.stop()
            if st.button("SCAN FIELD", type="primary", use_container_width=True):
                try:
                    from materials_map import interpret
                    from nos_morphological import morphological_nos
                    from scoring import cheap_descriptor_pvalues, cheap_score_from_image, fdr_decision, surrogate_null_test
                    from candidate_export import build_candidate, dumps_candidate
                    results = run_plugins(image, active)
                    try:
                        from scoring import adaptive_surrogate_null_test
                        mc = adaptive_surrogate_null_test(image, cheap_score_from_image(image), cheap_score_from_image, n_start=min(n_null, 24), n_expand=max(n_null, 80), seed=80)
                    except Exception:
                        mc = surrogate_null_test(image, cheap_score_from_image(image), cheap_score_from_image, n_simulations=n_null)
                    fdr = fdr_decision(cheap_descriptor_pvalues(image, n_simulations=max(12, n_null // 2), seed=80))
                    mc["fdr"] = fdr
                    materials = interpret(results, float(mc.get("z_score") or 0), float(mc.get("p_value") or 1), fdr_pass=bool(fdr.get("fdr_pass")))
                    nos = morphological_nos(results)
                    if prov["verdict"] != "usable_science" or prov.get("product_level") == "detector":
                        materials["is_candidate"] = False
                    st.write(materials.get("verdict"))
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("NOS v0", "%.2f" % nos.get("nos", 0))
                    emp = nos.get("empirical") or {}
                    if use_iso and emp.get("available"):
                        c2.metric("ISOFOREST", "%.3f" % float(emp.get("nos_isoforest") or 0))
                        c3.metric("OUTLIER", "SI" if emp.get("isoforest_outlier") else "NO")
                    c4.metric("FDR", "PASS" if fdr.get("fdr_pass") else "FAIL")
                    st.caption("p=%.4f  z=%.2f  p_crit=%.4f  fondo=%s" % (mc.get("p_value") or 1, mc.get("z_score") or 0, fdr.get("p_critical") or 1, emp.get("background")))
                    k41 = results.get("kolmogorov_1941") or {}
                    if "beta" in k41:
                        st.caption("Kolmogorov  beta=%.3f \u00b1 %s  R2=%s  k62=%s  iso=%s" % (k41.get("beta") or 0, k41.get("beta_se"), k41.get("r_squared"), k41.get("k62_kurtosis"), k41.get("isotropic_score")))
                    fib = results.get("fibonacci") or {}
                    if fib and "error" not in fib:
                        st.caption("Fibonacci  hits=%s  ratio=%s  score=%s" % (fib.get("n_phi_pairs"), fib.get("best_ratio"), fib.get("fibonacci_score")))
                    g = results.get("graph_morphology") or {}
                    if g and "error" not in g:
                        st.caption("Grafo  nodos=%s aristas=%s componentes=%s clustering=%.3f gap=%s" % (g.get("n_nodes"), g.get("n_edges"), g.get("n_components"), g.get("clustering") or 0, g.get("spectral_gap")))
                    fam = materials.get("family_scores") or {}
                    if fam:
                        st.bar_chart(fam)
                    if use_iso:
                        try:
                            from nos_empirical import tile_isolation_forest
                            tiles = tile_isolation_forest(image)
                            st.caption("ISO tiles  outliers=%s  pico=%s" % (tiles.get("n_outliers"), tiles.get("peak_tile")))
                            st.image(tiles["heatmap"], caption="rareza local IsolationForest", use_container_width=True, clamp=True)
                        except Exception as exc:
                            st.caption("iso tiles: %s" % exc)
                    st.download_button("DUMP JSON", dumps_candidate(build_candidate(meta["filename"], results, materials, prov, nos, mc, meta, src)), file_name="cms80_%s.json" % meta["filename"])
                except Exception as exc:
                    st.exception(exc)

with tab_cat:
    st.markdown("### Catalogo por mision")
    st.caption("Lista curada al instante. MAST solo cuando pides buscar.")
    mission = st.selectbox("Mision", ["JWST", "HST", "ROMAN", "HLSP"])
    rows = [{"objeto": n, "nota": w, "mision": mission} for n, w in CURATED.get(mission, [])]
    st.dataframe(rows, hide_index=True, use_container_width=True)
    pick = st.selectbox("Elegir objeto", [r["objeto"] for r in rows])
    if st.button("BUSCAR ESTE OBJETO EN MAST", use_container_width=True):
        try:
            from mast_client import search_observations
            with st.spinner("MAST..."):
                st.session_state["mast_obs"] = search_observations(pick, [mission])
            st.success("Observaciones listas. Vuelve a SCOUT.")
        except Exception as exc:
            st.error(str(exc))
    if st.button("AMPLIAR LISTA DESDE MAST (muestra)"):
        try:
            from mast_client import list_mission_targets
            extra = list_mission_targets(mission)
            st.dataframe(extra, hide_index=True, use_container_width=True)
        except Exception as exc:
            st.error(str(exc))

with tab_rub:
    st.markdown("### Rubrica de madurez")
    st.info("Un PNG de HubbleSite no es un material. Un FITS i2d/x1d + espectro de la misma region es el unico camino a mir_unknown.")
    st.table([
        {"Capa": "Ingenieria / UI / MAST", "Hoy": "7.5", "Meta": "8.5", "Nota": "Cloud arranca liviano; ciencia al SCAN"},
        {"Capa": "Integridad cientifica", "Hoy": "7.5", "Meta": "8.5-9", "Nota": "FDR, K62, beta\u00b1se, ERR/DQ en repo"},
        {"Capa": "Poder de descubrimiento", "Hoy": "3.5", "Meta": "5-6 techo ~7", "Nota": "IsolationForest sintetico hasta corpus i2d"},
        {"Capa": "Proteccion vs overclaim", "Hoy": "alta", "Meta": "alta", "Nota": "JSON con estados honestos"},
    ])
    st.markdown("**Bloque 1** FDR + Kolmogorov real — en codigo.")
    st.markdown("**Bloque 2** IsolationForest on-scan; corpus real con scripts/fit_nos.py.")
    st.markdown("**Bloque 3** scan_survey.py rankea. El radar no es el microscopio.")
