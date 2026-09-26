"""CMS-80 Cosmic Materials Scout. Terminal fosforosa."""
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
.stApp {
  background: repeating-linear-gradient(0deg, rgba(0,255,80,0.035) 0px, rgba(0,255,80,0.035) 1px, transparent 1px, transparent 3px),
              radial-gradient(ellipse at center, #051a0c 0%, #020803 70%);
  color: #b7ffc2;
}
.stApp::after {
  content: ""; pointer-events: none; position: fixed; inset: 0;
  background: linear-gradient(rgba(18,16,16,0) 50%, rgba(0,0,0,0.18) 50%);
  background-size: 100% 4px; z-index: 9999;
}
h1, h2, h3 { font-family: "VT323", monospace !important; color: #33ff66 !important; text-shadow: 0 0 8px #1aff55; }
.crt-banner { border: 1px solid #33ff66; box-shadow: 0 0 12px #145c28; padding: 0.85rem 1.1rem; margin: 0.4rem 0 1.1rem 0; color: #9dffb0; background: #031108; }
.crt-line { color: #33ff66; font-size: 1.35rem; }
.crt-sub { color: #7cbb88; font-size: 0.92rem; }
.blink { animation: blink 1.2s step-end infinite; }
@keyframes blink { 50% { opacity: 0; } }
[data-testid="stSidebar"] { background: #010604 !important; border-right: 1px solid #1a8f3c; }
.stButton>button { background: #031108 !important; color: #33ff66 !important; border: 1px solid #33ff66 !important; border-radius: 0 !important; }
[data-testid="stMetricValue"] { color: #33ff66 !important; font-family: "VT323", monospace !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="crt-banner">
  <div class="crt-line">CMS-80  COSMIC MATERIALS SCOUT  <span class="blink">&#9608;</span></div>
  <div class="crt-sub">SYS.READY | FITS ARCHIVE PREFERRED | MIR LIBRARY HANDOFF | NO PRESS JPEG DISCOVERY</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("`CMS-80 / CONFIG`")
source_url = st.sidebar.text_input("URL DE ORIGEN / FITS DIRECTO", value="")
st.sidebar.caption("Pega un .fits de MAST. PNG de galeria = GATE.REJECT")
mode = st.sidebar.selectbox("MODO", ["balanced", "explorer", "conservative"])
st.sidebar.markdown("`DESCRIPTORES`")
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
st.sidebar.caption("JPEG de galeria NASA = ilustracion. Candidato solo con FITS + espectro MIR.")


def _crt_layout(fig):
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor="#031108", font=dict(color=PHOSPHOR, family="Share Tech Mono"), margin=dict(t=40, b=30, l=40, r=20))
    fig.update_xaxes(gridcolor="#0d3a18", color=PHOSPHOR_DIM)
    fig.update_yaxes(gridcolor="#0d3a18", color=PHOSPHOR_DIM)
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
            return None, f"FITS > {max_mb:.0f} MB. Recorta o corre local."
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
                metadata = {
                    "filename": uploaded_file.name,
                    "format": "FITS",
                    "width": int(data.shape[1]),
                    "height": int(data.shape[0]),
                    "is_fits": True,
                    "instrument": str(header.get("INSTRUME", header.get("TELESCOP", ""))),
                    "filter": str(header.get("FILTER", header.get("FILTER1", ""))),
                }
                return data, metadata, None
        image = Image.open(uploaded_file)
        metadata = {
            "filename": uploaded_file.name,
            "format": image.format,
            "width": image.width,
            "height": image.height,
            "is_fits": False,
            "instrument": "",
            "filter": "",
        }
        if image.mode != "L":
            image = image.convert("L")
        return np.array(image, dtype=np.float32) / 255.0, metadata, None
    except Exception as exc:
        return None, None, str(exc)


def create_score_gauge(score: float, title: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        title={"text": title, "font": {"size": 16, "color": PHOSPHOR}},
        number={"font": {"color": PHOSPHOR}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": PHOSPHOR},
            "bar": {"color": PHOSPHOR},
            "bgcolor": "#031108",
            "bordercolor": PHOSPHOR_DIM,
            "steps": [
                {"range": [0, 40], "color": "#062010"},
                {"range": [40, 70], "color": "#0a3a16"},
                {"range": [70, 100], "color": "#125522"},
            ],
        },
    ))
    fig.update_layout(height=280, paper_bgcolor=BG, font={"color": PHOSPHOR}, margin=dict(t=40, b=10))
    return fig


def run_analysis(image, active):
    results = {}
    progress = st.progress(0)
    status = st.empty()
    class_steps = [
        ("fractal_base", "FRACTAL", "plugins.fractal_base", "FractalBase"),
        ("kolmogorov_1941", "P(k)", "plugins.kolmogorov_1941", "Kolmogorov1941"),
        ("lyapunov_stability", "ROSENSTEIN", "plugins.lyapunov_stability", "LyapunovStability"),
        ("persistent_homology", "TOPOLOGY", "plugins.persistent_homology", "PersistentHomology"),
        ("renormalization_group", "RG-BLOCK", "plugins.renormalization_group", "RenormalizationGroup"),
    ]
    n = max(len(active), 1)
    done = 0
    for key, label, modname, clsname in class_steps:
        if key not in active:
            continue
        status.text(f"> RUN {label}")
        mod = __import__(modname, fromlist=[clsname])
        results[key] = getattr(mod, clsname)().analyze(image)
        done += 1
        progress.progress(min(done / n, 1.0))
    if "anisotropy" in active:
        from plugins.anisotropy import calculate_anisotropy
        status.text("> RUN ANISO")
        results["anisotropy"] = calculate_anisotropy(image)
        done += 1
        progress.progress(min(done / n, 1.0))
    if "entropy" in active:
        from plugins.entropy import calculate_entropy
        status.text("> RUN ENTROPY")
        results["entropy"] = calculate_entropy(image)
        done += 1
        progress.progress(min(done / n, 1.0))
    if "periodicity" in active:
        from plugins.periodicity import analyze_periodicity
        status.text("> RUN FFT PEAKS")
        results["periodicity"] = analyze_periodicity(image)
        done += 1
        progress.progress(min(done / n, 1.0))
    status.empty()
    progress.empty()
    return results


def parse_spectrum_csv(file) -> dict:
    import pandas as pd
    df = pd.read_csv(file)
    if len(df.columns) < 2:
        return {"status": "error", "kind": None, "xy": None, "notes": "CSV necesita 2 columnas x,y"}
    cols = [c.lower() for c in df.columns]
    xcol, ycol = df.columns[0], df.columns[1]
    for i, c in enumerate(cols):
        if c in {"x", "wavenumber", "wavelength", "um", "cm-1", "lambda"}:
            xcol = df.columns[i]
        if c in {"y", "flux", "intensity", "absorbance", "i"}:
            ycol = df.columns[i]
    xy = [{"x": float(a), "y": float(b)} for a, b in zip(df[xcol], df[ycol]) if np.isfinite(a) and np.isfinite(b)]
    kind = "ftir_or_mir" if "cm" in str(xcol).lower() or "wave" in str(xcol).lower() else "unknown"
    return {"status": "loaded", "kind": kind, "xy": xy[:8000], "notes": f"{len(xy)} puntos desde {file.name}"}


uploaded_file = st.file_uploader("LOAD FIELD  [FITS / FIT / PNG / JPG / TIFF]", type=["jpg", "jpeg", "png", "tiff", "tif", "fits", "fit"])
if uploaded_file is None and source_url.strip().lower().endswith((".fits", ".fit", ".fits.gz")):
    if st.button("PULL FITS FROM URL", use_container_width=True):
        with st.spinner("> HTTP GET"):
            try:
                remote, err = fetch_url_file(source_url)
            except Exception as exc:
                remote, err = None, str(exc)
        if err:
            st.error(f"NET.ERR  {err}")
        else:
            uploaded_file = remote
            st.session_state["remote_fits"] = remote
if uploaded_file is None and "remote_fits" in st.session_state:
    uploaded_file = st.session_state["remote_fits"]

if uploaded_file is None:
    st.markdown("""
```
CMS-80 BOOT
  [1] FITS de MAST / ALMA / ESO      = canal ciencia
  [2] JPEG HubbleSite / Webb gallery = ilustracion
  [3] Scout morfologico              = familia + NOS
  [4] Libreria MIR                   = espectro x,y de la MISMA region
  [5] Frontera                       = mir_unknown + FITS de archivo
NOMBRE UN MINERAL DESDE UN PNG = SYS.REFUSE
```
""")
else:
    with st.spinner("> MOUNT DATASET"):
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
        if prov["verdict"] == "reject_for_discovery":
            st.error("GATE.REJECT  fuente tipo prensa/JPEG. Explorar si, publicar no.")
        elif prov["verdict"] == "exploratory_only":
            st.warning("GATE.WARN  exploratorio. No cuenta como frontera.")
        else:
            st.success("GATE.OK  FITS de archivo. Canal ciencia abierto.")
        for r in prov["reasons"]:
            st.caption(f"> {r}")
        fig, ax = plt.subplots(figsize=(6.2, 6.2), facecolor=BG)
        ax.set_facecolor(BG)
        ax.imshow(image, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
        allow_candidate = prov["verdict"] != "reject_for_discovery"
        run_label = "SCAN FIELD" if allow_candidate else "SCAN FIELD  (exploratorio)"
        if st.button(run_label, type="primary", use_container_width=True):
            with st.spinner("> DESCRIPTORS"):
                plugin_results = run_analysis(image, active_plugins)
            global_score = compute_global_score(plugin_results, mode)
            observed_cheap = cheap_score_from_image(image)
            mc_results = surrogate_null_test(image, observed_cheap, cheap_score_from_image, n_simulations=n_null)
            materials = interpret(plugin_results, structure_z=float(mc_results.get("z_score") or 0.0), p_value=float(mc_results.get("p_value") or 1.0))
            nos = morphological_nos(plugin_results)
            if not allow_candidate:
                materials["is_candidate"] = False
                materials["verdict"] = "Fuente no publicable. " + str(materials.get("verdict", ""))
            st.markdown("## `HYPOTHESIS`")
            if materials["is_candidate"] and allow_candidate:
                st.success(materials["verdict"])
            elif materials.get("instrument_warning") or not allow_candidate:
                st.warning(materials["verdict"])
            else:
                st.info(materials["verdict"])
            col1, col2 = st.columns([1, 2])
            with col1:
                st.plotly_chart(create_score_gauge(1.0 - nos["nos"], "RAREZA  (1-NOS)"), use_container_width=True)
                st.metric("NOS MORFOLOGICO", f"{nos['nos']:.2f}")
                st.caption(f"nearest= {nos['nearest_class']}")
            with col2:
                st.markdown(f"`FAMILIA`  {materials['dominant_label']}")
                st.markdown(f"`LAB`      {materials['lab_analog']}")
                st.markdown(f"`CIELO`    {materials['sky_analog']}")
                st.markdown(f"`NEXT`     {materials['followup']}")
                st.caption(f"p={mc_results['p_value']:.3f} {mc_results['stars']}  z={float(mc_results.get('z_score') or 0):.2f}  n={mc_results['n_simulations']}")
                fam = materials["family_scores"]
                figb = px.bar(x=list(fam.keys()), y=list(fam.values()), labels={"x": "familia", "y": "peso"}, title="MEZCLA")
                figb.update_traces(marker_color=PHOSPHOR)
                st.plotly_chart(_crt_layout(figb), use_container_width=True)
            st.markdown("## `MIR HANDOFF  ->  spectral-identifier-v1`")
            spec_file = st.file_uploader("OPCIONAL: CSV espectro de la MISMA region  [x,y]", type=["csv", "txt"], key="mir_csv")
            spectrum_block = {"status": "missing", "kind": None, "xy": None, "notes": "Sin espectro MIR/FTIR/Raman de la region."}
            if spec_file is not None:
                try:
                    spectrum_block = parse_spectrum_csv(spec_file)
                    st.success(f"SPECTRUM.OK  {spectrum_block['notes']}")
                except Exception as exc:
                    st.error(f"SPECTRUM.ERR  {exc}")
            card = build_candidate(
                filename=metadata["filename"],
                plugin_results=plugin_results,
                materials=materials,
                provenance=prov,
                nos=nos,
                monte_carlo=mc_results,
                metadata=metadata,
                source_url=source_url,
            )
            card["spectrum"] = spectrum_block
            card["spectral_identifier_handoff"] = {
                "target": "WilmerGaspar/spectral-identifier-v1",
                "endpoint": "/api/identify/json",
                "ready": spectrum_block.get("status") == "loaded",
                "reason": "Listo para POST del xy a la libreria MIR/FTIR." if spectrum_block.get("status") == "loaded" else "Sin xy espectral no hay identificacion de material.",
            }
            json_txt = dumps_candidate(card)
            st.download_button("DUMP CANDIDATE.JSON", data=json_txt, file_name=f"cms80_{metadata['filename']}.json", mime="application/json", use_container_width=True)
            with st.expander("VIEW JSON"):
                st.code(json_txt, language="json")
            payload = {
                "timestamp": datetime.now().isoformat(),
                "filename": metadata.get("filename"),
                "mode": mode,
                "global_score": global_score,
                "monte_carlo": mc_results,
                "plugin_results": plugin_results,
                "materials": materials,
            }
            with st.expander("INFORME"):
                st.markdown(generate_report(payload))
            st.markdown("## `CHANNELS`")
            tabs = st.tabs([name.replace("_", " ").upper() for name in active_plugins])
            for i, plugin_name in enumerate(active_plugins):
                with tabs[i]:
                    result = plugin_results.get(plugin_name, {})
                    if plugin_name == "fractal_base":
                        a, b, c = st.columns(3)
                        a.metric("D0", f"{result.get('d0', 0):.3f}")
                        b.metric("D1", f"{result.get('d1', 0):.3f}")
                        c.metric("D2", f"{result.get('d2', 0):.3f}")
                    elif plugin_name == "kolmogorov_1941":
                        a, b, c = st.columns(3)
                        a.metric("BETA", f"{result.get('beta', 0):.3f}")
                        b.metric("R2", f"{result.get('r_squared', 0):.3f}")
                        c.metric("INTER", f"{result.get('intermittency_factor', 0):.3f}")
                    elif plugin_name == "periodicity":
                        a, b, c = st.columns(3)
                        a.metric("SCORE", f"{result.get('periodicity_score', 0):.3f}")
                        b.metric("PEAKS", f"{result.get('n_significant_peaks', 0)}")
                        c.metric("LATTICE", str(result.get("lattice_hint", "-")))
                        if result.get("likely_instrument_artifact"):
                            st.warning(result.get("note", "ARTIFACT?"))
                    elif plugin_name == "lyapunov_stability":
                        st.metric("LAMBDA", f"{result.get('max_lyapunov', 0):.4f}")
                    elif plugin_name == "persistent_homology":
                        a, b = st.columns(2)
                        a.metric("B0", f"{result.get('betti_0', 0)}")
                        b.metric("B1", f"{result.get('betti_1', 0)}")
                    elif plugin_name == "renormalization_group":
                        st.metric("XI", f"{result.get('correlation_length', 0):.2f}")
                    elif plugin_name == "anisotropy":
                        st.metric("ANISO", f"{result.get('anisotropy_index', 0):.3f}")
                    elif plugin_name == "entropy":
                        st.metric("H", f"{result.get('shannon_entropy_bits', 0):.3f}")
                    else:
                        st.json(result)
