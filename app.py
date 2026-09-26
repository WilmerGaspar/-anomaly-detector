"""CMS-80 Cosmic Materials Scout. Terminal fosforosa + MAST API."""
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
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
PHOSPHOR_DIM = "#1a8f3c"
BG = "#020803"

st.set_page_config(page_title="CMS-80 | COSMIC MATERIALS SCOUT", page_icon="■", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=VT323&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] { font-family: "Share Tech Mono", "VT323", ui-monospace, monospace !important; }
.stApp { background: radial-gradient(ellipse at center, #051a0c 0%, #020803 70%); color: #b7ffc2; }
.crt-banner { border: 1px solid #33ff66; padding: 0.85rem 1.1rem; margin: 0.4rem 0 1.1rem 0; color: #9dffb0; background: #031108; }
.crt-line { color: #33ff66; font-size: 1.35rem; }
.blink { animation: blink 1.2s step-end infinite; }
@keyframes blink { 50% { opacity: 0; } }
[data-testid="stSidebar"] { background: #010604 !important; }
.stButton>button { background: #031108 !important; color: #33ff66 !important; border: 1px solid #33ff66 !important; border-radius: 0 !important; }
[data-testid="stMetricValue"] { color: #33ff66 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="crt-banner"><div class="crt-line">CMS-80  COSMIC MATERIALS SCOUT  <span class="blink">&#9608;</span></div>
<div>MAST API | FITS ARCHIVE | MIR HANDOFF | NO PRESS JPEG</div></div>""", unsafe_allow_html=True)

st.sidebar.markdown("`CMS-80 / CONFIG`")
source_url = st.sidebar.text_input("URL DE ORIGEN / FITS DIRECTO", value="")
mode = st.sidebar.selectbox("MODO", ["balanced", "explorer", "conservative"])
plugins = {
    "fractal_base": st.sidebar.checkbox("FRACTAL D0/D1/D2", value=True),
    "kolmogorov_1941": st.sidebar.checkbox("CASCADA / P(k)", value=True),
    "periodicity": st.sidebar.checkbox("RED / PICOS FFT", value=True),
    "anisotropy": st.sidebar.checkbox("FILAMENTO / ANISO", value=True),
    "persistent_homology": st.sidebar.checkbox("TOPOLOGIA", value=True),
    "renormalization_group": st.sidebar.checkbox("COARSE-GRAIN", value=True),
    "lyapunov_stability": st.sidebar.checkbox("ROSENSTEIN", value=True),
    "entropy": st.sidebar.checkbox("SHANNON", value=True),
}
active_plugins = [k for k, v in plugins.items() if v]
n_null = st.sidebar.slider("SUBROGADOS NULOS", 10, 80, 24, 2)


def _crt_layout(fig):
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor="#031108", font=dict(color=PHOSPHOR), margin=dict(t=40, b=30, l=40, r=20))
    return fig


class _MemFile:
    def __init__(self, name, data):
        self.name = name
        self._buf = __import__("io").BytesIO(data)
    def read(self, *a, **k):
        return self._buf.read(*a, **k)
    def seek(self, *a, **k):
        return self._buf.seek(*a, **k)


def fetch_url_file(url: str, max_mb: float = 80.0):
    import requests
    url = (url or "").strip()
    if not url.startswith(("http://", "https://")):
        return None, "URL invalida"
    name = url.split("?")[0].rstrip("/").split("/")[-1] or "remote.fits"
    r = requests.get(url, timeout=60, stream=True)
    r.raise_for_status()
    cap = int(max_mb * 1024 * 1024)
    chunks, total = [], 0
    for blk in r.iter_content(1024 * 64):
        total += len(blk)
        if total > cap:
            return None, f"FITS > {max_mb:.0f} MB"
        chunks.append(blk)
    return _MemFile(name, b"".join(chunks)), None


@st.cache_data
def load_and_preprocess_image(uploaded_file):
    try:
        name = uploaded_file.name.lower()
        if name.endswith((".fits", ".fit", ".fits.gz")):
            from astropy.io import fits
            with fits.open(uploaded_file) as hdul:
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
                    return None, None, "NO SCI DATA IN FITS"
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
                metadata = {"filename": uploaded_file.name, "format": "FITS", "width": int(data.shape[1]), "height": int(data.shape[0]), "is_fits": True, "instrument": str(header.get("INSTRUME", header.get("TELESCOP", ""))), "filter": str(header.get("FILTER", header.get("FILTER1", "")))}
                return data, metadata, None
        image = Image.open(uploaded_file)
        metadata = {"filename": uploaded_file.name, "format": image.format, "width": image.width, "height": image.height, "is_fits": False, "instrument": "", "filter": ""}
        if image.mode != "L":
            image = image.convert("L")
        return np.array(image, dtype=np.float32) / 255.0, metadata, None
    except Exception as exc:
        return None, None, str(exc)


def create_score_gauge(score: float, title: str):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=score * 100, title={"text": title, "font": {"size": 16, "color": PHOSPHOR}}, number={"font": {"color": PHOSPHOR}}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": PHOSPHOR}, "bgcolor": "#031108"}))
    fig.update_layout(height=280, paper_bgcolor=BG, font={"color": PHOSPHOR})
    return fig


def run_analysis(image, active):
    results = {}
    class_steps = [("fractal_base", "plugins.fractal_base", "FractalBase"), ("kolmogorov_1941", "plugins.kolmogorov_1941", "Kolmogorov1941"), ("lyapunov_stability", "plugins.lyapunov_stability", "LyapunovStability"), ("persistent_homology", "plugins.persistent_homology", "PersistentHomology"), ("renormalization_group", "plugins.renormalization_group", "RenormalizationGroup")]
    for key, modname, clsname in class_steps:
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
    return results


st.markdown("`MAST API`")
mc1, mc2, mc3 = st.columns([2, 1, 1])
with mc1:
    mast_target = st.text_input("OBJETO", value="NGC 7023")
with mc2:
    mast_missions = st.multiselect("MISION", ["HST", "JWST", "HLSP"], default=["HST", "JWST"])
with mc3:
    do_mast = st.button("QUERY MAST", use_container_width=True)

if do_mast and mast_target.strip():
    try:
        from mast_client import search_observations
        with st.spinner("> MAST.query_object"):
            st.session_state["mast_obs"] = search_observations(mast_target, mast_missions)
        if not st.session_state.get("mast_obs"):
            st.warning("MAST.EMPTY  desmarca TESS; usa HST o JWST.")
    except Exception as exc:
        st.error(f"MAST.ERR  {exc}")

obs_rows = st.session_state.get("mast_obs") or []
if obs_rows:
    labels = [f"{r['mission']} | {r['instrument']} | {r['filters']} | {r['obs_id']}" for r in obs_rows]
    pick = st.selectbox("SELECCIONA OBSERVACION", labels)
    chosen = obs_rows[labels.index(pick)]
    if st.button("LIST FITS DE ESTA OBS", use_container_width=True):
        try:
            from mast_client import list_fits_products
            with st.spinner("> get_product_list"):
                st.session_state["mast_prods"] = list_fits_products(chosen["obsid"])
        except Exception as exc:
            st.error(f"MAST.ERR  {exc}")

prods = st.session_state.get("mast_prods") or []
if prods:
    plabels = [f"{p['filename']}  ({p['size_mb']:.1f} MB)" if p.get("size_mb") else p["filename"] for p in prods]
    p_pick = st.selectbox("SELECCIONA FITS", plabels)
    prod = prods[plabels.index(p_pick)]
    if st.button("PULL FITS FROM MAST", use_container_width=True):
        try:
            from mast_client import download_product
            with st.spinner("> download_file"):
                blob = download_product(prod["uri"], prod["filename"])
            st.session_state["remote_fits"] = _MemFile(prod["filename"], blob)
            source_url = prod.get("uri") or source_url
            st.success(f"MAST.OK  {prod['filename']}")
        except Exception as exc:
            st.error(f"MAST.ERR  {exc}")

uploaded_file = st.file_uploader("LOAD FIELD", type=["jpg", "jpeg", "png", "tiff", "tif", "fits", "fit"])
if uploaded_file is None and source_url.strip().lower().endswith((".fits", ".fit", ".fits.gz")):
    if st.button("PULL FITS FROM URL", use_container_width=True):
        remote, err = (None, None)
        try:
            remote, err = fetch_url_file(source_url)
        except Exception as exc:
            err = str(exc)
        if err:
            st.error(f"NET.ERR  {err}")
        else:
            uploaded_file = remote
            st.session_state["remote_fits"] = remote
if uploaded_file is None and "remote_fits" in st.session_state:
    uploaded_file = st.session_state["remote_fits"]

if uploaded_file is None:
    st.code("CMS-80: QUERY MAST -> elige HST/JWST -> LIST FITS -> PULL. No uses TESS aqui.")
else:
    image, metadata, error = load_and_preprocess_image(uploaded_file)
    if error:
        st.error(f"SYS.ERR  {error}")
    else:
        prov = assess_provenance(metadata["filename"], metadata, source_url)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("FILE", metadata["filename"][:18])
        c2.metric("SIZE", f"{metadata['width']}x{metadata['height']}")
        c3.metric("TRUST", f"{prov['trust_score']:.2f}")
        c4.metric("GATE", prov["verdict"])
        fig, ax = plt.subplots(figsize=(6, 6), facecolor=BG)
        ax.set_facecolor(BG)
        ax.imshow(image, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
        allow_candidate = prov["verdict"] != "reject_for_discovery"
        if st.button("SCAN FIELD" if allow_candidate else "SCAN FIELD (exploratorio)", type="primary", use_container_width=True):
            plugin_results = run_analysis(image, active_plugins)
            global_score = compute_global_score(plugin_results, mode)
            mc_results = surrogate_null_test(image, cheap_score_from_image(image), cheap_score_from_image, n_simulations=n_null)
            materials = interpret(plugin_results, structure_z=float(mc_results.get("z_score") or 0.0), p_value=float(mc_results.get("p_value") or 1.0))
            nos = morphological_nos(plugin_results)
            if not allow_candidate:
                materials["is_candidate"] = False
            st.markdown("## `HYPOTHESIS`")
            st.write(materials["verdict"])
            col1, col2 = st.columns([1, 2])
            with col1:
                st.plotly_chart(create_score_gauge(1.0 - nos["nos"], "RAREZA (1-NOS)"), use_container_width=True)
                st.metric("NOS", f"{nos['nos']:.2f}")
            with col2:
                st.write(f"FAMILIA  {materials['dominant_label']}")
                st.write(f"LAB      {materials['lab_analog']}")
                fam = materials["family_scores"]
                figb = px.bar(x=list(fam.keys()), y=list(fam.values()), title="MEZCLA")
                figb.update_traces(marker_color=PHOSPHOR)
                st.plotly_chart(_crt_layout(figb), use_container_width=True)
            card = build_candidate(metadata["filename"], plugin_results, materials, prov, nos, mc_results, metadata, source_url)
            st.download_button("DUMP CANDIDATE.JSON", data=dumps_candidate(card), file_name=f"cms80_{metadata['filename']}.json", mime="application/json")
