"""CMS-80. Sin cache sobre archivos. INICIAR + Fibonacci."""
from datetime import datetime
import io

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import streamlit as st
from PIL import Image

from candidate_export import build_candidate, dumps_candidate
from materials_map import interpret
from nos_morphological import morphological_nos
from provenance import assess_provenance
from report_generator import generate_report
from scoring import cheap_score_from_image, compute_global_score, surrogate_null_test

PHOSPHOR = "#33ff66"
BG = "#020803"

st.set_page_config(page_title="CMS-80 | COSMIC MATERIALS SCOUT", page_icon="\u25a0", layout="wide")
st.markdown("""
<style>
.stApp { background:#020803; color:#b7ffc2; }
.stButton>button { background:#031108 !important; color:#33ff66 !important; border:1px solid #33ff66 !important; border-radius:0 !important; }
[data-testid="stMetricValue"] { color:#33ff66 !important; }
[data-testid="stSidebar"] { background:#010604 !important; }
</style>
""", unsafe_allow_html=True)
st.markdown("## CMS-80  COSMIC MATERIALS SCOUT")
st.caption("MAST Hubble / Webb / Roman  \u00b7  preview + FITS  \u00b7  INICIAR")

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
}
active_plugins = [k for k, v in plugins.items() if v]
n_null = st.sidebar.slider("NULOS", 10, 80, 24, 2)


def load_from_bytes(name, raw):
    try:
        buf = io.BytesIO(raw)
        low = name.lower()
        if low.endswith((".fits", ".fit", ".fits.gz")):
            from astropy.io import fits
            with fits.open(buf) as hdul:
                data = None
                for hdu in hdul:
                    if getattr(hdu, "data", None) is not None and getattr(hdu, "name", "") == "SCI":
                        data = hdu.data
                        break
                if data is None:
                    for hdu in hdul:
                        if getattr(hdu, "data", None) is not None:
                            data = hdu.data
                            break
                if data is None:
                    return None, None, "NO SCI DATA"
                data = np.nan_to_num(np.array(data, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                if data.ndim > 2:
                    data = np.squeeze(data)
                    if data.ndim > 2:
                        data = data[0]
                p2, p98 = np.percentile(data, [2, 98])
                if p98 > p2:
                    data = (data - p2) / (p98 - p2)
                data = np.clip(data, 0.0, 1.0)
                header = hdul[0].header
                meta = {"filename": name, "format": "FITS", "width": int(data.shape[1]), "height": int(data.shape[0]), "is_fits": True, "instrument": str(header.get("INSTRUME", header.get("TELESCOP", ""))), "filter": str(header.get("FILTER", header.get("FILTER1", "")))}
                return data, meta, None
        image = Image.open(buf)
        if image.mode != "L":
            image = image.convert("L")
        arr = np.array(image, dtype=np.float32) / 255.0
        meta = {"filename": name, "format": image.format, "width": image.width, "height": image.height, "is_fits": False, "instrument": "", "filter": ""}
        return arr, meta, None
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
    return results


st.markdown("### DATOS  \u2014  Hubble / Webb / Roman")
d1, d2, d3 = st.columns([2, 2, 1])
with d1:
    target = st.text_input("Objeto o coordenadas", value="NGC 7023")
with d2:
    missions = st.multiselect("Misiones", ["HST", "JWST", "ROMAN", "HLSP"], default=["HST", "JWST"])
with d3:
    go_q = st.button("BUSCAR EN MAST", use_container_width=True)

if go_q and target.strip():
    try:
        from mast_client import search_observations
        with st.spinner("consultando MAST..."):
            st.session_state["mast_obs"] = search_observations(target, missions)
        st.session_state.pop("mast_prods", None)
        if not st.session_state["mast_obs"]:
            st.warning("Nada en esas misiones. Roman puede no tener campo publico aun.")
    except Exception as exc:
        st.error("MAST.ERR  " + str(exc))

obs_rows = st.session_state.get("mast_obs") or []
if obs_rows:
    labels = ["%s | %s | %s | %s | %s" % (r["mission"], r["instrument"], r["filters"], r["target"], r["obs_id"]) for r in obs_rows]
    pick = st.selectbox("Observaciones encontradas", labels)
    chosen = obs_rows[labels.index(pick)]
    left, right = st.columns(2)
    with left:
        st.json({"mision": chosen["mission"], "instrumento": chosen["instrument"], "filtro": chosen["filters"], "objetivo": chosen["target"], "RA": chosen.get("s_ra"), "Dec": chosen.get("s_dec"), "exptime": chosen.get("t_exptime"), "obs_id": chosen["obs_id"]})
    with right:
        if chosen.get("jpeg_url"):
            st.image(chosen["jpeg_url"], caption="preview MAST")
        else:
            st.caption("Sin preview JPEG en esta fila.")
    if st.button("VER FITS DE ESTA FOTO", use_container_width=True):
        try:
            from mast_client import list_fits_products
            with st.spinner("listando productos..."):
                st.session_state["mast_prods"] = list_fits_products(chosen["obsid"])
        except Exception as exc:
            st.error("MAST.ERR  " + str(exc))

prods = st.session_state.get("mast_prods") or []
if prods:
    plabels = []
    for p in prods:
        mb = p.get("size_mb")
        plabels.append("%s (%s)" % (p["filename"], "?.? MB" if mb is None else ("%.1f MB" % mb)))
    p_pick = st.selectbox("Producto FITS (<80 MB)", plabels)
    prod = prods[plabels.index(p_pick)]
    if st.button("CARGAR FITS", use_container_width=True):
        try:
            from mast_client import download_product
            with st.spinner("bajando FITS..."):
                blob = download_product(prod["uri"], prod["filename"])
            st.session_state["field_name"] = prod["filename"]
            st.session_state["field_bytes"] = blob
            st.session_state["field_url"] = prod.get("uri") or ""
            st.session_state["started"] = False
            st.success(prod["filename"])
        except Exception as exc:
            st.error("MAST.ERR  " + str(exc))

st.markdown("### O carga local")
up = st.file_uploader("FITS / imagen", type=["jpg", "jpeg", "png", "tiff", "tif", "fits", "fit"])
if up is not None:
    st.session_state["field_name"] = up.name
    st.session_state["field_bytes"] = up.getvalue()
    st.session_state["field_url"] = source_url
    st.session_state["started"] = False

if source_url.strip().lower().endswith((".fits", ".fit", ".fits.gz")) and st.button("BAJAR URL"):
    import requests
    try:
        r = requests.get(source_url.strip(), timeout=60)
        r.raise_for_status()
        st.session_state["field_name"] = source_url.split("?")[0].rstrip("/").split("/")[-1]
        st.session_state["field_bytes"] = r.content
        st.session_state["field_url"] = source_url
        st.session_state["started"] = False
    except Exception as exc:
        st.error(str(exc))

name = st.session_state.get("field_name")
raw = st.session_state.get("field_bytes")
src = st.session_state.get("field_url") or source_url

if not raw:
    st.info("Busca un objeto, carga el FITS y pulsa INICIAR.")
else:
    st.success("ARCHIVO EN BUFFER: %s  (%d KB)" % (name, len(raw) // 1024))
    if st.button("INICIAR", type="primary", use_container_width=True):
        st.session_state["started"] = True
    if not st.session_state.get("started"):
        st.caption("Pulsa INICIAR para montar el campo.")
        st.stop()
    image, metadata, error = load_from_bytes(name, raw)
    if error:
        st.error(error)
    else:
        prov = assess_provenance(metadata["filename"], metadata, src)
        a, b, c, d = st.columns(4)
        a.metric("FILE", metadata["filename"][:18])
        b.metric("SIZE", "%sx%s" % (metadata["width"], metadata["height"]))
        c.metric("TRUST", "%.2f" % prov["trust_score"])
        d.metric("GATE", prov["verdict"])
        fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor=BG)
        ax.set_facecolor(BG)
        ax.imshow(image, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
        if st.button("SCAN FIELD", type="primary", use_container_width=True):
            plugin_results = run_analysis(image, active_plugins)
            mc = surrogate_null_test(image, cheap_score_from_image(image), cheap_score_from_image, n_simulations=n_null)
            materials = interpret(plugin_results, float(mc.get("z_score") or 0.0), float(mc.get("p_value") or 1.0))
            nos = morphological_nos(plugin_results)
            if prov["verdict"] == "reject_for_discovery":
                materials["is_candidate"] = False
            st.write(materials["verdict"])
            st.metric("NOS", "%.2f" % nos["nos"])
            fib = plugin_results.get("fibonacci") or {}
            if fib:
                st.metric("PHI HITS", str(fib.get("n_phi_pairs", 0)))
                st.caption("ratio=%.3f  err=%.3f  score=%.2f" % (fib.get("best_ratio") or 0, fib.get("phi_error") or 1, fib.get("fibonacci_score") or 0))
            st.write(materials["dominant_label"] + "  |  " + materials["lab_analog"])
            fam = materials["family_scores"]
            figb = px.bar(x=list(fam.keys()), y=list(fam.values()), title="MEZCLA")
            figb.update_traces(marker_color=PHOSPHOR)
            figb.update_layout(paper_bgcolor=BG, plot_bgcolor="#031108", font=dict(color=PHOSPHOR))
            st.plotly_chart(figb, use_container_width=True)
            card = build_candidate(metadata["filename"], plugin_results, materials, prov, nos, mc, metadata, src)
            st.download_button("DUMP JSON", dumps_candidate(card), file_name="cms80_%s.json" % metadata["filename"])
