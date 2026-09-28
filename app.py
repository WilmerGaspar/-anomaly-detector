import streamlit as st

st.set_page_config(page_title="CMS-80", page_icon="\u25a0", layout="wide")
st.markdown("""<style>
.stApp { background:#020803; color:#b7ffc2; }
.stButton>button { background:#031108 !important; color:#33ff66 !important; border:1px solid #33ff66 !important; border-radius:0 !important; }
[data-testid="stMetricValue"] { color:#33ff66 !important; }
[data-testid="stSidebar"] { background:#010604 !important; }
</style>""", unsafe_allow_html=True)
st.markdown("## CMS-80  COSMIC MATERIALS SCOUT")
st.caption("elige un objeto · mira su investigacion · carga i2d · SCAN")

NOTES = {
    "M16": "Pilares de la creacion. Polvo + fotoevaporacion. Busca i2d NIRCam/WFC3, no JPG de prensa.",
    "M51": "Espiral + acompanante. Control de filamentos y brazos, no rareza extrema.",
    "Crab Nebula": "Filamentos + cascada. Control positivo: el pipeline DEBE ver estructura.",
    "Hoag Object": "Anillo + nucleo. Gold set: si sale featureless, el NOS esta mal.",
    "Red Square Nebula": "Simetria casi cristalina. Cuidado con spikes del telescopio.",
    "NGC 7023": "Nube de polvo. Campo de control (MIRI F2550W ya corrido).",
    "Stephan's Quintet": "Grupo interactuando. Choques, no un material nuevo.",
    "SMACS 0723": "Cúmulo lente. Morfologia de arco, no espectro de grano.",
    "Jupiter": "Planeta. Atmosfera, no ISM. Util para probar el loader.",
    "PDS 70": "Disco protoplanetario. Candidato a followup IR.",
    "LMC": "Campo Roman/HST amplio.",
    "SMC": "Campo amplio.",
    "Andromeda": "M31. Demasiado grande: elige un recorte i2d, no el mosaic entero.",
    "HUDF": "Campo profundo. Galaxias lejanas, no polvo local.",
    "GOODS-S": "Survey. Elige un i2d concreto.",
    "COSMOS": "Survey. Elige un i2d concreto.",
}
CURATED = {
    "JWST": ["NGC 7023", "M16", "Stephan's Quintet", "SMACS 0723", "Jupiter", "PDS 70"],
    "HST": ["M16", "M51", "Crab Nebula", "Hoag Object", "Red Square Nebula", "NGC 7023"],
    "ROMAN": ["LMC", "SMC", "Andromeda"],
    "HLSP": ["HUDF", "GOODS-S", "COSMOS"],
}

if "object" not in st.session_state:
    st.session_state["object"] = "NGC 7023"
if "mission" not in st.session_state:
    st.session_state["mission"] = "JWST"

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

def run_plugins(image, active):
    out = {}
    class_steps = [("fractal_base", "plugins.fractal_base", "FractalBase"), ("kolmogorov_1941", "plugins.kolmogorov_1941", "Kolmogorov1941"), ("lyapunov_stability", "plugins.lyapunov_stability", "LyapunovStability"), ("persistent_homology", "plugins.persistent_homology", "PersistentHomology"), ("renormalization_group", "plugins.renormalization_group", "RenormalizationGroup")]
    fn_steps = [("anisotropy", "plugins.anisotropy", "calculate_anisotropy"), ("entropy", "plugins.entropy", "calculate_entropy"), ("periodicity", "plugins.periodicity", "analyze_periodicity"), ("fibonacci", "plugins.fibonacci", "analyze_fibonacci"), ("graph_morphology", "plugins.graph_morphology", "analyze_graph")]
    for key, mod, cls in class_steps:
        if key in active:
            try:
                m = __import__(mod, fromlist=[cls]); out[key] = getattr(m, cls)().analyze(image)
            except Exception as exc:
                out[key] = {"error": str(exc)}
    for key, mod, fn in fn_steps:
        if key in active:
            try:
                m = __import__(mod, fromlist=[fn]); out[key] = getattr(m, fn)(image)
            except Exception as exc:
                out[key] = {"error": str(exc)}
    return out

tab_cat, tab_scout, tab_rub = st.tabs(["1. OBJETO", "2. SCOUT", "3. RUBRICA"])

with tab_cat:
    st.markdown("### Selecciona objeto e investigacion")
    mission = st.selectbox("Mision", ["JWST", "HST", "ROMAN", "HLSP"], key="mission")
    options = CURATED.get(mission, ["NGC 7023"])
    obj = st.selectbox("Objeto fotografiado", options, key="object")
    st.info("%s \u2014 %s" % (obj, NOTES.get(obj, "Busca un i2d/x1d de archivo, no un PNG de prensa.")))
    c1, c2 = st.columns(2)
    with c1:
        load_inv = st.button("CARGAR INVESTIGACION MAST", type="primary", use_container_width=True)
    with c2:
        st.caption("Timeout 45s. Si falla, Upload en SCOUT.")
    if load_inv:
        try:
            from mast_client import search_observations
            with st.spinner("MAST %s / %s ..." % (mission, obj)):
                st.session_state["mast_obs"] = search_observations(obj, [mission])
            st.session_state.pop("mast_prods", None)
            st.session_state.pop("mast_pick", None)
        except Exception as exc:
            st.error(str(exc))
    obs_rows = st.session_state.get("mast_obs") or []
    if obs_rows:
        st.success("%d observaciones de %s" % (len(obs_rows), obj))
        st.dataframe(
            [{"mision": r.get("mission"), "instrumento": r.get("instrument"), "filtro": r.get("filters"), "objeto": r.get("target"), "obs_id": r.get("obs_id")} for r in obs_rows],
            hide_index=True, use_container_width=True,
        )
        labels = ["%s | %s | %s | %s" % (r.get("mission"), r.get("instrument"), r.get("filters"), r.get("obs_id")) for r in obs_rows]
        pick = st.selectbox("Elegir observacion", labels, key="obs_pick")
        chosen = obs_rows[labels.index(pick)]
        st.session_state["mast_pick"] = chosen
        if chosen.get("jpeg_url"):
            st.image(chosen["jpeg_url"], caption="preview %s" % chosen.get("obs_id"))
        if st.button("LISTAR FITS DE ESTA OBSERVACION", use_container_width=True):
            try:
                from mast_client import list_fits_products
                with st.spinner("productos FITS..."):
                    st.session_state["mast_prods"] = list_fits_products(chosen["obsid"])
            except Exception as exc:
                st.error(str(exc))
    else:
        st.caption("Aun no hay filas MAST. Pulsa CARGAR INVESTIGACION o pasa a SCOUT y sube un FITS.")
    prods = st.session_state.get("mast_prods") or []
    if prods:
        filtro = st.radio("Producto", ["recomendados (i2d/drz/x1d)", "todos", "crudos"], horizontal=True, key="prod_filter")
        shown = prods
        if filtro.startswith("recomendados"):
            shown = [p for p in prods if any(k in p["filename"].lower() for k in ("i2d", "drz", "drc", "x1d", "s3d"))] or prods
        elif filtro.startswith("crudos"):
            shown = [p for p in prods if any(k in p["filename"].lower() for k in ("uncal", "rate"))] or prods
        st.dataframe([{"archivo": p["filename"], "MB": p.get("size_mb"), "nota": p.get("hint")} for p in shown], hide_index=True, use_container_width=True)
        plabels = ["%s | %s" % (p["filename"], p.get("hint") or "") for p in shown]
        ip = st.selectbox("Selecciona FITS", range(len(plabels)), format_func=lambda i: plabels[i], key="fits_pick")
        if st.button("CARGAR ESTE FITS AL SCOUT", type="primary", use_container_width=True):
            try:
                from mast_client import download_product
                p = shown[ip]
                with st.spinner("descarga..."):
                    st.session_state["field_bytes"] = download_product(p["uri"], p["filename"])
                st.session_state["field_name"] = p["filename"]
                st.session_state["field_url"] = p.get("uri") or ""
                st.session_state["started"] = False
                st.success("FITS en buffer. Abre pestaña 2. SCOUT y pulsa INICIAR.")
            except Exception as exc:
                st.error(str(exc))

with tab_scout:
    st.markdown("### Scout de %s (%s)" % (st.session_state.get("object"), st.session_state.get("mission")))
    up = st.file_uploader("Upload local i2d/x1d", type=["jpg", "jpeg", "png", "tif", "tiff", "fits", "fit"], key="local_up")
    if up is not None:
        st.session_state["field_name"] = up.name
        st.session_state["field_bytes"] = up.getvalue()
        st.session_state["started"] = False
    name = st.session_state.get("field_name")
    raw = st.session_state.get("field_bytes")
    src = st.session_state.get("field_url") or source_url
    if not raw:
        st.warning("No hay FITS en buffer. Ve a 1. OBJETO, carga investigacion y un i2d, o sube un archivo aqui.")
    else:
        st.success("BUFFER %s" % name)
        if st.button("INICIAR", type="primary", use_container_width=True):
            st.session_state["started"] = True
        if st.session_state.get("started"):
            try:
                import io, numpy as np
                from PIL import Image
                from astropy.io import fits
                from provenance import assess_provenance
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
                            st.error("NO SCI DATA")
                        else:
                            image = np.nan_to_num(np.array(data, dtype=np.float32), nan=0.0)
                            if image.ndim > 2:
                                image = np.squeeze(image)
                                if image.ndim > 2:
                                    image = image[0]
                            p2, p98 = np.percentile(image, [2, 98])
                            image = np.clip((image - p2) / (p98 - p2), 0, 1) if p98 > p2 else np.clip(image, 0, 1)
                            header = hdul[0].header
                            meta = {"filename": name, "format": "FITS", "width": int(image.shape[1]), "height": int(image.shape[0]), "is_fits": True, "instrument": str(header.get("INSTRUME", header.get("TELESCOP", ""))), "filter": str(header.get("FILTER", header.get("FILTER1", "")))}
                            st.session_state["image"] = image
                            st.session_state["meta"] = meta
                else:
                    im = Image.open(buf).convert("L")
                    image = np.array(im, dtype=np.float32) / 255.0
                    meta = {"filename": name, "format": im.format, "width": im.width, "height": im.height, "is_fits": False, "instrument": "", "filter": ""}
                    st.session_state["image"] = image
                    st.session_state["meta"] = meta
                meta = st.session_state.get("meta") or {"filename": name, "width": 0, "height": 0}
                image = st.session_state.get("image")
                prov = assess_provenance(str(meta.get("filename")), meta, src)
                st.session_state["prov"] = prov
                a,b,c,d = st.columns(4)
                a.metric("FILE", str(name)[:22]); b.metric("SIZE", "%sx%s" % (meta.get("width"), meta.get("height")))
                c.metric("TRUST", "%.2f" % prov["trust_score"]); d.metric("GATE", prov["verdict"])
                if image is not None:
                    st.image(image, caption=str(name), use_container_width=True, clamp=True)
            except Exception as exc:
                st.exception(exc)
            if st.button("SCAN FIELD", type="primary", use_container_width=True):
                try:
                    image = st.session_state.get("image")
                    meta = st.session_state.get("meta") or {"filename": name}
                    prov = st.session_state.get("prov") or {"verdict": "unknown", "trust_score": 0}
                    if image is None:
                        st.error("Pulsa INICIAR primero.")
                    else:
                        from materials_map import interpret
                        from nos_morphological import morphological_nos
                        from scoring import cheap_descriptor_pvalues, cheap_score_from_image, fdr_decision, surrogate_null_test
                        from candidate_export import build_candidate, dumps_candidate
                        results = run_plugins(image, active)
                        mc = surrogate_null_test(image, cheap_score_from_image(image), cheap_score_from_image, n_simulations=n_null)
                        fdr = fdr_decision(cheap_descriptor_pvalues(image, n_simulations=max(12, n_null // 2), seed=80))
                        mc["fdr"] = fdr
                        materials = interpret(results, float(mc.get("z_score") or 0), float(mc.get("p_value") or 1), fdr_pass=bool(fdr.get("fdr_pass")))
                        nos = morphological_nos(results)
                        if prov.get("verdict") != "usable_science" or prov.get("product_level") == "detector":
                            materials["is_candidate"] = False
                        st.write(materials.get("verdict"))
                        x1,x2,x3,x4 = st.columns(4)
                        x1.metric("NOS v0", "%.2f" % nos.get("nos", 0))
                        emp = nos.get("empirical") or {}
                        if use_iso and emp.get("available"):
                            x2.metric("ISOFOREST", "%.3f" % float(emp.get("nos_isoforest") or 0))
                            x3.metric("OUTLIER", "SI" if emp.get("isoforest_outlier") else "NO")
                        x4.metric("FDR", "PASS" if fdr.get("fdr_pass") else "FAIL")
                        k41 = results.get("kolmogorov_1941") or {}
                        if "beta" in k41:
                            st.caption("beta=%.3f \u00b1 %s  k62=%s" % (k41.get("beta") or 0, k41.get("beta_se"), k41.get("k62_kurtosis")))
                        fam = materials.get("family_scores") or {}
                        if fam:
                            st.bar_chart(fam)
                        st.download_button("DUMP JSON", dumps_candidate(build_candidate(meta.get("filename"), results, materials, prov, nos, mc, meta, src)), file_name="cms80_%s.json" % meta.get("filename"))
                except Exception as exc:
                    st.exception(exc)

with tab_rub:
    st.info("Un PNG de HubbleSite no es un material. FITS i2d/x1d + espectro de la misma region.")
    st.table([
        {"Capa": "Ingenieria", "Hoy": "7.5"},
        {"Capa": "Integridad", "Hoy": "7.5"},
        {"Capa": "Descubrimiento", "Hoy": "3.5"},
    ])
