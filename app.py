"""CMS-80: scout + catalogo + IsolationForest."""
import io
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from candidate_export import build_candidate, dumps_candidate
from materials_map import interpret
from nos_morphological import morphological_nos
from provenance import assess_provenance
from scoring import cheap_descriptor_pvalues, cheap_score_from_image, fdr_decision

try:
    from scoring import adaptive_surrogate_null_test as _null
except Exception:
    from scoring import surrogate_null_test as _null

PHOSPHOR = "#33ff66"
BG = "#020803"
st.set_page_config(page_title="CMS-80", page_icon="\u25a0", layout="wide")
st.markdown("""<style>
.stApp { background:#020803; color:#b7ffc2; }
.stButton>button { background:#031108 !important; color:#33ff66 !important; border:1px solid #33ff66 !important; border-radius:0 !important; }
[data-testid="stMetricValue"] { color:#33ff66 !important; }
[data-testid="stSidebar"] { background:#010604 !important; }
</style>""", unsafe_allow_html=True)
st.markdown("## CMS-80  COSMIC MATERIALS SCOUT")
st.caption("i2d/x1d \u00b7 IsolationForest \u00b7 FDR \u00b7 catalogo por mision")

st.sidebar.markdown("`CONFIG`")
source_url = st.sidebar.text_input("URL FITS DIRECTO", value="")
mode = st.sidebar.selectbox("MODO", ["balanced", "explorer", "conservative"])
plugins = {
    "fractal_base": st.sidebar.checkbox("FRACTAL", value=True),
    "kolmogorov_1941": st.sidebar.checkbox("P(k)", value=True),
    "periodicity": st.sidebar.checkbox("FFT", value=True),
    "anisotropy": st.sidebar.checkbox("ANISO", value=True),
    "persistent_homology": st.sidebar.checkbox("TOPO", value=True),
    "renormalization_group": st.sidebar.checkbox("RG", value=True),
    "lyapunov_stability": st.sidebar.checkbox("ROSENSTEIN", value=True),
    "entropy": st.sidebar.checkbox("SHANNON", value=True),
    "fibonacci": st.sidebar.checkbox("FIBONACCI / PHI", value=True),
    "graph_morphology": st.sidebar.checkbox("GRAFOS", value=True),
}
active_plugins = [k for k, v in plugins.items() if v]
n_null = st.sidebar.slider("NULOS", 10, 80, 24, 2)

def load_from_bytes(name, raw):
    try:
        buf = io.BytesIO(raw)
        low = name.lower()
        if low.endswith((".fits", ".fit", ".fits.gz")):
            from astropy.io import fits
            try:
                from fits_quality import apply_weights, extract_layers, good_pixel_mask, snr_weights, stretch_masked, summarize_quality
                from instrument_mask import apply_spike_mask
                use_q = True
            except Exception:
                use_q = False
            with fits.open(buf) as hdul:
                if use_q:
                    layers = extract_layers(hdul)
                    if layers["sci"] is None:
                        return None, None, "NO SCI DATA"
                    mask = good_pixel_mask(layers["sci"], layers["dq"], layers["err"])
                    data = apply_weights(stretch_masked(layers["sci"], mask), snr_weights(layers["sci"], layers["err"], mask))
                    data, spikes = apply_spike_mask(data)
                    q = summarize_quality(mask, layers["dq"], layers["err"])
                    q["spikes_masked"] = int(spikes.sum())
                else:
                    data = None
                    for hdu in hdul:
                        if getattr(hdu, "data", None) is not None:
                            data = hdu.data
                            if str(getattr(hdu, "name", "")) == "SCI":
                                break
                    if data is None:
                        return None, None, "NO SCI DATA"
                    data = np.nan_to_num(np.array(data, dtype=np.float32), nan=0.0)
                    if data.ndim > 2:
                        data = np.squeeze(data)
                        if data.ndim > 2: data = data[0]
                    p2, p98 = np.percentile(data, [2, 98])
                    if p98 > p2: data = (data - p2) / (p98 - p2)
                    data = np.clip(data, 0.0, 1.0)
                    q = {}
                header = hdul[0].header
                meta = {"filename": name, "format": "FITS", "width": int(data.shape[1]), "height": int(data.shape[0]), "is_fits": True, "instrument": str(header.get("INSTRUME", header.get("TELESCOP", ""))), "filter": str(header.get("FILTER", header.get("FILTER1", ""))), "quality": q}
                return data, meta, None
        image = Image.open(buf)
        if image.mode != "L": image = image.convert("L")
        return np.array(image, dtype=np.float32)/255.0, {"filename": name, "format": image.format, "width": image.width, "height": image.height, "is_fits": False, "instrument": "", "filter": ""}, None
    except Exception as exc:
        return None, None, str(exc)

def run_analysis(image, active):
    results = {}
    steps = [("fractal_base", "plugins.fractal_base", "FractalBase"), ("kolmogorov_1941", "plugins.kolmogorov_1941", "Kolmogorov1941"), ("lyapunov_stability", "plugins.lyapunov_stability", "LyapunovStability"), ("persistent_homology", "plugins.persistent_homology", "PersistentHomology"), ("renormalization_group", "plugins.renormalization_group", "RenormalizationGroup")]
    for key, modname, clsname in steps:
        if key in active:
            mod = __import__(modname, fromlist=[clsname])
            results[key] = getattr(mod, clsname)().analyze(image)
    if "anisotropy" in active:
        from plugins.anisotropy import calculate_anisotropy
        results["anisotropy"] = calculate_anisotropy(image)
    if "entropy" in active:
        from plugins.entropy import calculate_entropy
        results["entropy"] = calculate_entropy(image)
    if "periodicity" in active:
        from plugins.periodicity import analyze_periodicity
        results["periodicity"] = analyze_periodicity(image)
    if "fibonacci" in active:
        from plugins.fibonacci import analyze_fibonacci
        results["fibonacci"] = analyze_fibonacci(image)
    if "graph_morphology" in active:
        from plugins.graph_morphology import analyze_graph
        results["graph_morphology"] = analyze_graph(image)
    return results

tab_scout, tab_cat, tab_rub = st.tabs(["SCOUT", "CATALOGO MISION", "RUBRICA"])

with tab_scout:
    d1, d2, d3 = st.columns([2, 2, 1])
    with d1: target = st.text_input("Objeto", value="NGC 7023")
    with d2: missions = st.multiselect("Misiones", ["HST", "JWST", "ROMAN", "HLSP"], default=["HST", "JWST"])
    with d3: go_q = st.button("BUSCAR EN MAST", use_container_width=True)
    if go_q and target.strip():
        try:
            from mast_client import search_observations
            with st.spinner("MAST (timeout 45s, 2 intentos)..."):
                st.session_state["mast_obs"] = search_observations(target, missions)
            st.session_state.pop("mast_prods", None)
            if not st.session_state["mast_obs"]:
                st.warning("Sin filas. Prueba CATALOGO MISION o carga local.")
        except Exception as exc:
            st.error(str(exc))
            st.info("MAST a veces no responde. Usa CATALOGO o Upload de un i2d.")
    obs_rows = st.session_state.get("mast_obs") or []
    if obs_rows:
        labels = ["%s | %s | %s | %s | %s" % (r["mission"], r["instrument"], r["filters"], r.get("target"), r["obs_id"]) for r in obs_rows]
        pick = st.selectbox("Observacion", labels)
        chosen = obs_rows[labels.index(pick)]
        if chosen.get("jpeg_url"): st.image(chosen["jpeg_url"], caption="preview")
        if st.button("LISTAR TODOS LOS FITS", use_container_width=True):
            try:
                from mast_client import list_fits_products
                with st.spinner("listando FITS..."):
                    st.session_state["mast_prods"] = list_fits_products(chosen["obsid"])
            except Exception as exc:
                st.error(str(exc))
    prods = st.session_state.get("mast_prods") or []
    if prods:
        filtro = st.radio("Filtro", ["recomendados (i2d/drz/x1d)", "todos", "crudos"], horizontal=True)
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
                    blob = download_product(prod["uri"], prod["filename"])
                st.session_state["field_name"] = prod["filename"]
                st.session_state["field_bytes"] = blob
                st.session_state["field_url"] = prod.get("uri") or ""
                st.session_state["started"] = False
            except Exception as exc:
                st.error(str(exc))
    up = st.file_uploader("O carga local", type=["jpg", "jpeg", "png", "tiff", "tif", "fits", "fit"])
    if up is not None:
        st.session_state["field_name"] = up.name
        st.session_state["field_bytes"] = up.getvalue()
        st.session_state["started"] = False
    name = st.session_state.get("field_name")
    raw = st.session_state.get("field_bytes")
    src = st.session_state.get("field_url") or source_url
    if not raw:
        st.info("Si MAST falla: CATALOGO MISION o Upload. Luego INICIAR.")
    else:
        st.success("BUFFER %s" % name)
        if st.button("INICIAR", type="primary", use_container_width=True):
            st.session_state["started"] = True
        if st.session_state.get("started"):
            image, metadata, error = load_from_bytes(name, raw)
            if error:
                st.error(error)
            else:
                prov = assess_provenance(metadata["filename"], metadata, src)
                a,b,c,d = st.columns(4)
                a.metric("FILE", metadata["filename"][:22]); b.metric("SIZE", "%sx%s" % (metadata["width"], metadata["height"]))
                c.metric("TRUST", "%.2f" % prov["trust_score"]); d.metric("GATE", prov["verdict"])
                fig, ax = plt.subplots(figsize=(5,5), facecolor=BG); ax.set_facecolor(BG); ax.imshow(image, cmap="gray"); ax.axis("off"); st.pyplot(fig); plt.close(fig)
                if st.button("SCAN FIELD", type="primary", use_container_width=True):
                    plugin_results = run_analysis(image, active_plugins)
                    try:
                        mc = _null(image, cheap_score_from_image(image), cheap_score_from_image, n_start=min(n_null,24), n_expand=max(n_null,80), seed=80)
                    except TypeError:
                        mc = _null(image, cheap_score_from_image(image), cheap_score_from_image, n_simulations=n_null)
                    fdr = fdr_decision(cheap_descriptor_pvalues(image, n_simulations=max(12, n_null//2), seed=80))
                    mc["fdr"] = fdr
                    materials = interpret(plugin_results, float(mc.get("z_score") or 0), float(mc.get("p_value") or 1), fdr_pass=bool(fdr.get("fdr_pass")))
                    nos = morphological_nos(plugin_results)
                    if prov["verdict"] != "usable_science" or prov.get("product_level") == "detector":
                        materials["is_candidate"] = False
                    st.write(materials["verdict"])
                    c1, c2, c3 = st.columns(3)
                    c1.metric("NOS v0", "%.2f" % nos["nos"])
                    emp = nos.get("empirical") or {}
                    if emp.get("available"):
                        c2.metric("ISOFOREST", "%.3f" % float(emp.get("nos_isoforest") or 0))
                        c3.metric("OUTLIER", "SI" if emp.get("isoforest_outlier") else "NO")
                        st.caption("fondo=%s  %s" % (emp.get("background"), emp.get("note")))
                    else:
                        c2.metric("ISOFOREST", "n/a")
                    try:
                        from nos_empirical import tile_isolation_forest
                        tiles = tile_isolation_forest(image)
                        figt = px.imshow(tiles["heatmap"], title="Rareza IsolationForest tiles  outliers=%s  pico=%s" % (tiles["n_outliers"], tiles["peak_tile"]), color_continuous_scale="Turbo")
                        figt.update_layout(paper_bgcolor=BG, font=dict(color=PHOSPHOR))
                        st.plotly_chart(figt, use_container_width=True)
                    except Exception as exc:
                        st.caption("iso tiles: %s" % exc)
                    fam = materials.get("family_scores") or {}
                    figb = px.bar(x=list(fam.keys()), y=list(fam.values()), title="MEZCLA DE FAMILIAS")
                    figb.update_traces(marker_color=PHOSPHOR)
                    figb.update_layout(paper_bgcolor=BG, plot_bgcolor="#031108", font=dict(color=PHOSPHOR))
                    st.plotly_chart(figb, use_container_width=True)
                    k41 = plugin_results.get("kolmogorov_1941") or {}
                    if k41.get("k_values") and k41.get("spectrum"):
                        figk = go.Figure()
                        figk.add_trace(go.Scatter(x=k41["k_values"], y=k41["spectrum"], mode="lines+markers", line=dict(color=PHOSPHOR)))
                        figk.update_layout(title="P(k)  beta=%.3f se=%s" % (k41.get("beta") or 0, k41.get("beta_se")), xaxis_type="log", yaxis_type="log", paper_bgcolor=BG, plot_bgcolor="#031108", font=dict(color=PHOSPHOR))
                        st.plotly_chart(figk, use_container_width=True)
                    g = plugin_results.get("graph_morphology") or {}
                    if g:
                        st.caption("GRAFO  nodos=%s aristas=%s componentes=%s clustering=%.3f gap=%s" % (g.get("n_nodes"), g.get("n_edges"), g.get("n_components"), g.get("clustering") or 0, g.get("spectral_gap")))
                    st.download_button("DUMP JSON", dumps_candidate(build_candidate(metadata["filename"], plugin_results, materials, prov, nos, mc, metadata, src)), file_name="cms80_%s.json" % metadata["filename"])

with tab_cat:
    st.markdown("### Catalogo por mision")
    st.caption("Curado + muestra MAST. No es el archivo entero.")
    mission = st.selectbox("Mision", ["JWST", "HST", "ROMAN", "HLSP"])
    if st.button("LISTAR OBJETOS DE ESTA MISION", use_container_width=True):
        try:
            from mast_client import list_mission_targets
            with st.spinner("consultando catalogo..."):
                st.session_state["cat_rows"] = list_mission_targets(mission)
        except Exception as exc:
            st.error(str(exc))
    rows = st.session_state.get("cat_rows") or []
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)
        names = [r.get("target") or "" for r in rows if r.get("source") != "error"]
        if names:
            chosen_t = st.selectbox("Elegir objeto y buscar observaciones", names)
            if st.button("BUSCAR ESTE OBJETO", use_container_width=True):
                try:
                    from mast_client import search_observations
                    with st.spinner("MAST..."):
                        st.session_state["mast_obs"] = search_observations(chosen_t, [mission])
                    st.success("Listo. Vuelve a SCOUT.")
                except Exception as exc:
                    st.error(str(exc))

with tab_rub:
    st.markdown("### Rubrica")
    st.info("IsolationForest ya puntua rareza. Sin corpus i2d usa fondo sintetico; no lo cites como descubrimiento.")
    st.table([
        {"Capa": "Ingenieria / UI / MAST", "Hoy": "7.5", "Meta": "8.5"},
        {"Capa": "Integridad", "Hoy": "7.5", "Meta": "8.5-9"},
        {"Capa": "Descubrimiento", "Hoy": "3.5", "Meta": "5-6 techo ~7"},
        {"Capa": "Overclaim", "Hoy": "alta", "Meta": "alta"},
    ])
