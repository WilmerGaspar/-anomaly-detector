"""CMS-80 Cosmic Materials Scout — interfaz Streamlit.

Flujo unico: 1) fuente de datos con procedencia visible, 2) imagen y region,
3) sonificacion comparativa, 4) analisis con nulo IAAFT + FDR.
Requiere streamlit >= 1.35 (seleccion de filas en st.dataframe).
"""
import io
import json

import numpy as np
import streamlit as st

st.set_page_config(page_title="CMS-80", page_icon="\u25a0", layout="wide")
st.markdown("""<style>
.stApp { background:#020803; color:#b7ffc2; }
.stButton>button, .stDownloadButton>button { background:#031108 !important; color:#33ff66 !important; border:1px solid #33ff66 !important; border-radius:0 !important; }
.stButton>button:disabled { color:#2a5a35 !important; border-color:#2a5a35 !important; }
[data-testid="stMetricValue"] { color:#33ff66 !important; }
[data-testid="stSidebar"] { background:#010604 !important; }
.cms-prov { border-left:2px solid #33ff66; padding:0.3rem 0.8rem; margin:0.4rem 0; font-size:0.92rem; }
</style>""", unsafe_allow_html=True)

from mast_client import CURATED, MAST_ACK, MAX_CLOUD_MB, NOTES  # noqa: E402

SEED = 80
MAX_ANALYSIS_SIDE = 2048
PREVIEW_SIDE = 900

st.markdown("## CMS-80 Cosmic Materials Scout")
st.caption("Busca organización espacial tipo material en imágenes científicas (FITS) y la compara con un nulo "
           "que conserva espectro e histograma. No identifica compuestos: eso exige espectroscopía.")

# ---------------------------------------------------------------- ajustes avanzados
with st.sidebar:
    st.markdown("**Ajustes avanzados**")
    n_null = st.select_slider("Subrogados del nulo", options=[49, 99, 199, 499], value=99,
                              help="p mínimo posible = 1/(n+1). Con 99, p ≥ 0.01.")
    st.caption("p mínimo posible: %.4f" % (1.0 / (n_null + 1)))
    st.markdown("**Descriptores**")
    plugins = {
        "fractal_base": st.checkbox("Fractal (D0, Dq, lacunaridad)", True),
        "kolmogorov_1941": st.checkbox("Espectro P(k) e intermitencia", True),
        "periodicity": st.checkbox("Periodicidad FFT", True),
        "anisotropy": st.checkbox("Anisotropía", True),
        "persistent_homology": st.checkbox("Topología", True),
        "renormalization_group": st.checkbox("Coarse-graining (RG)", True),
        "entropy": st.checkbox("Entropía de Shannon", True),
        "graph_morphology": st.checkbox("Grafos", True),
    }
    with st.expander("Exploratorios (no usar para conclusiones)"):
        plugins["lyapunov_stability"] = st.checkbox("Rosenstein (proxy espacial)", False)
        plugins["fibonacci"] = st.checkbox("Fibonacci / phi", False)
    active = [k for k, v in plugins.items() if v]


def run_plugins(image, active):
    out = {}
    class_steps = [("fractal_base", "plugins.fractal_base", "FractalBase"), ("kolmogorov_1941", "plugins.kolmogorov_1941", "Kolmogorov1941"),
                   ("lyapunov_stability", "plugins.lyapunov_stability", "LyapunovStability"), ("persistent_homology", "plugins.persistent_homology", "PersistentHomology"),
                   ("renormalization_group", "plugins.renormalization_group", "RenormalizationGroup")]
    fn_steps = [("anisotropy", "plugins.anisotropy", "calculate_anisotropy"), ("entropy", "plugins.entropy", "calculate_entropy"),
                ("periodicity", "plugins.periodicity", "analyze_periodicity"), ("fibonacci", "plugins.fibonacci", "analyze_fibonacci"),
                ("graph_morphology", "plugins.graph_morphology", "analyze_graph")]
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
                out[key] = getattr(m, fn)(image)
            except Exception as exc:
                out[key] = {"error": str(exc)}
    return out


def set_field(data, name, source):
    """Guarda el archivo elegido y su procedencia; invalida resultados previos."""
    st.session_state["field_bytes"] = data
    st.session_state["field_name"] = name
    st.session_state["field_source"] = source
    for k in ("parsed", "parsed_key", "results"):
        st.session_state.pop(k, None)


# ---------------------------------------------------------------- carga de imagen

def list_image_hdus(raw):
    from astropy.io import fits
    out = []
    with fits.open(io.BytesIO(raw), memmap=False) as hdul:
        for i, hdu in enumerate(hdul):
            if not getattr(hdu, "is_image", False) or hdu.data is None:
                continue
            shape = tuple(hdu.data.shape)
            ndim = len([s for s in shape if s > 1])
            if ndim == 2:
                out.append({"index": i, "name": hdu.name or "PRIMARY", "shape": shape})
    return out


def parse_field(raw, name, hdu_index=None):
    low = str(name).lower()
    if low.endswith((".fits", ".fit", ".fits.gz")):
        from astropy.io import fits
        with fits.open(io.BytesIO(raw), memmap=False) as hdul:
            hdu = hdul[hdu_index]
            data = np.squeeze(np.asarray(hdu.data, dtype=np.float32))
            h0, h1 = hdul[0].header, hdu.header

            def hk(*keys):
                for k in keys:
                    for h in (h0, h1):
                        if k in h and str(h[k]).strip():
                            return str(h[k]).strip()
                return ""
            meta = {"filename": name, "format": "FITS", "is_fits": True, "hdu": "%s[%d]" % (hdu.name or "PRIMARY", hdu_index),
                    "telescope": hk("TELESCOP"), "instrument": hk("INSTRUME"), "filter": hk("FILTER", "FILTER1", "FILTER2"),
                    "target_header": hk("TARGNAME", "TARGPROP", "OBJECT"), "date_obs": hk("DATE-OBS", "DATE-BEG"),
                    "program": hk("PROGRAM", "PROPOSID"), "bunit": hk("BUNIT")}
    else:
        from PIL import Image
        im = Image.open(io.BytesIO(raw))
        fmt = im.format
        data = np.asarray(im.convert("L"), dtype=np.float32)
        meta = {"filename": name, "format": fmt, "is_fits": False, "hdu": "", "instrument": "", "filter": ""}
    meta["width"], meta["height"] = int(data.shape[1]), int(data.shape[0])
    return data, meta


def finite_bbox(data):
    finite = np.isfinite(data)
    rows, cols = np.where(finite.any(axis=1))[0], np.where(finite.any(axis=0))[0]
    if rows.size == 0:
        return None
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def stretch(a, lo=2, hi=98):
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros_like(a, dtype=np.float32), (0.0, 0.0)
    p_lo, p_hi = np.percentile(finite, [lo, hi])
    out = np.where(np.isfinite(a), a, np.median(finite))
    out = np.clip((out - p_lo) / (p_hi - p_lo), 0, 1) if p_hi > p_lo else np.zeros_like(out)
    return out.astype(np.float32), (float(p_lo), float(p_hi))


def preview_with_box(data, y0, x0, side):
    from PIL import Image, ImageDraw
    step = max(1, int(np.ceil(max(data.shape) / PREVIEW_SIDE)))
    small, _ = stretch(data[::step, ::step])
    im = Image.fromarray((small * 255).astype(np.uint8)).convert("RGB")
    d = ImageDraw.Draw(im)
    d.rectangle([x0 // step, y0 // step, (x0 + side) // step, (y0 + side) // step], outline=(51, 255, 102), width=2)
    return im


def sonify(img, sr=22050, dur=8.0, fmin=220.0, fmax=1760.0, ref=None):
    """Sonificacion (no es sonido real): columnas de izquierda a derecha.
    Tono = brillo medio de la columna (escala log fmin-fmax).
    Volumen = contraste de la columna (desviacion estandar).
    `ref` fija la escala con otra imagen para que la comparacion sea justa."""
    a = np.asarray(img, dtype=np.float64)
    base = a if ref is None else np.concatenate([a, np.asarray(ref, dtype=np.float64)], axis=1)
    mean_c, std_c = a.mean(axis=0), a.std(axis=0)
    m_lo, m_hi = base.mean(axis=0).min(), base.mean(axis=0).max()
    s_hi = base.std(axis=0).max() or 1.0
    n = int(sr * dur)
    x = np.linspace(0, len(mean_c) - 1, n)
    b = np.interp(x, np.arange(len(mean_c)), (mean_c - m_lo) / ((m_hi - m_lo) or 1.0))
    amp = np.interp(x, np.arange(len(std_c)), std_c / s_hi)
    freq = fmin * (fmax / fmin) ** np.clip(b, 0, 1)
    y = 0.35 * (0.15 + 0.85 * amp) * np.sin(2 * np.pi * np.cumsum(freq) / sr)
    fade = np.minimum(1.0, np.minimum(np.arange(n), np.arange(n)[::-1]) / (0.02 * sr))
    return (y * fade).astype(np.float32)


# ================================================================ 1. FUENTE DE DATOS
st.markdown("### 1. Fuente de datos")
source_kind = st.radio("Origen", ["Archivo MAST (STScI)", "Archivo propio", "URL directa"], horizontal=True, key="source_kind")

if source_kind == "Archivo MAST (STScI)":
    c1, c2, c3 = st.columns([1, 2, 1])
    mission = c1.selectbox("Misión", list(CURATED.keys()), key="mission")
    obj_pick = c2.selectbox("Objeto", CURATED[mission] + ["Otro…"], key="obj_pick")
    obj = c2.text_input("Nombre del objeto (resuelto por MAST)", key="obj_free") if obj_pick == "Otro…" else obj_pick
    radius = c3.number_input("Radio (grados)", 0.01, 0.5, 0.12, 0.01, key="radius")
    if obj in NOTES:
        st.caption(NOTES[obj])
    if st.button("Buscar en MAST", type="primary", disabled=not obj):
        try:
            from mast_client import search_observations_detailed
            with st.spinner("Consultando MAST…"):
                st.session_state["search"] = search_observations_detailed(obj, [mission], radius_deg=radius)
            st.session_state.pop("prods", None)
        except Exception as exc:
            st.error(str(exc))

    res = st.session_state.get("search")
    if res:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Observaciones en MAST", res["n_total"])
        m2.metric("De %s" % "/".join(res["missions"]), res["n_mission"])
        m3.metric("Imágenes", res["n_images"])
        m4.metric("En la tabla", res["n_shown"])
        st.caption("Búsqueda: '%s', radio %.2f°. %s" % (res["target"], res["radius_deg"],
                   "Se muestran las primeras %d imágenes." % res["n_shown"] if res["truncated"] else "Se muestran todas las imágenes."))
        rows = res["rows"]
        if not rows:
            st.warning("MAST no devolvió imágenes de esta misión para este objeto. Prueba otra misión, otro nombre o un radio mayor.")
        else:
            table = [{"misión": r["mission"], "instrumento": r["instrument"], "filtro": r["filters"], "objeto": r["target"],
                      "fecha": r["date_obs"], "exp (s)": r["t_exptime"], "programa": r["proposal_id"], "PI": r["proposal_pi"],
                      "obs_id": r["obs_id"]} for r in rows]
            st.markdown("Haz clic en una fila para ver su procedencia.")
            ev = st.dataframe(table, hide_index=True, use_container_width=True, on_select="rerun",
                              selection_mode="single-row", key="obs_table")
            sel = ev.selection.rows
            if sel:
                r = rows[sel[0]]
                left, right = st.columns([1, 2])
                if r["jpeg_url"]:
                    left.image(r["jpeg_url"], caption="Vista previa de MAST: %s" % r["obs_id"])
                else:
                    left.info("Esta observación no tiene vista previa en MAST.")
                right.markdown(
                    "<div class='cms-prov'><b>Procedencia</b><br>"
                    "Archivo: MAST · STScI<br>Misión / instrumento: %s / %s<br>Filtro: %s<br>"
                    "Programa: %s · PI: %s<br>Título: %s<br>Fecha: %s · Exposición: %s s<br>"
                    "Coordenadas: RA %s, Dec %s<br>Nivel de calibración: %s · Derechos: %s<br>obs_id: %s</div>"
                    % (r["mission"], r["instrument"], r["filters"] or "—", r["proposal_id"] or "—", r["proposal_pi"] or "—",
                       r["obs_title"] or "—", r["date_obs"] or "—", r["t_exptime"] if r["t_exptime"] is not None else "—",
                       r["s_ra"], r["s_dec"], r["calib_level"] if r["calib_level"] is not None else "—", r["data_rights"] or "—", r["obs_id"]),
                    unsafe_allow_html=True)
                if r["data_url"]:
                    right.markdown("[Producto principal en MAST](%s)" % r["data_url"])
                if right.button("Listar archivos FITS de esta observación", type="primary"):
                    try:
                        from mast_client import list_fits_products
                        with st.spinner("Listando archivos…"):
                            st.session_state["prods"] = list_fits_products(r["obsid"])
                        st.session_state["prods_obs"] = r
                    except Exception as exc:
                        st.error(str(exc))

    prods = st.session_state.get("prods")
    if prods is not None and st.session_state.get("search"):
        obs_r = st.session_state.get("prods_obs") or {}
        st.markdown("**Archivos FITS de %s**" % obs_r.get("obs_id", ""))
        only_img = st.toggle("Solo imágenes 2D analizables", True, key="only_img")
        shown = [p for p in prods if p["is_image"]] if only_img else prods
        st.caption("Mostrando %d de %d archivos FITS." % (len(shown), len(prods)))
        if not shown:
            st.warning("Esta observación no tiene imágenes 2D. Desactiva el filtro para ver todo o elige otra fila.")
        else:
            ptable = [{"archivo": p["filename"], "MB": p["size_mb"], "imagen 2D": "sí" if p["is_image"] else "no", "nota": p["hint"]} for p in shown]
            pev = st.dataframe(ptable, hide_index=True, use_container_width=True, on_select="rerun",
                               selection_mode="single-row", key="prod_table")
            if pev.selection.rows:
                p = shown[pev.selection.rows[0]]
                reason = ("Pesa más de %d MB." % int(MAX_CLOUD_MB)) if p["too_big"] else ("No es una imagen 2D." if not p["is_image"] else "")
                if reason:
                    st.warning(reason)
                if p["download_url"]:
                    st.caption("Descarga directa: %s" % p["download_url"])
                if st.button("Cargar %s" % p["filename"], type="primary", disabled=bool(reason)):
                    try:
                        from mast_client import download_product
                        with st.spinner("Descargando %s…" % p["filename"]):
                            data = download_product(p["uri"], p["filename"], size_mb=p["size_mb"])
                        src = dict(obs_r)
                        src.update({"archive": "MAST", "product": p["filename"], "uri": p["uri"],
                                    "download_url": p["download_url"], "acknowledgement": MAST_ACK})
                        set_field(data, p["filename"], src)
                    except Exception as exc:
                        st.error(str(exc))

elif source_kind == "Archivo propio":
    up = st.file_uploader("FITS (recomendado) o imagen", type=["fits", "fit", "png", "jpg", "jpeg", "tif", "tiff"], key="local_up")
    if up is not None and up.name != st.session_state.get("field_name"):
        set_field(up.getvalue(), up.name, {"archive": "archivo local del usuario", "product": up.name})
    st.caption("Indica en tu informe de dónde sale el archivo: el sistema no puede verificarlo.")

else:
    url = st.text_input("URL de un archivo FITS", key="direct_url")
    if st.button("Descargar desde la URL", type="primary", disabled=not url):
        try:
            from mast_client import download_url
            with st.spinner("Descargando…"):
                data = download_url(url)
            set_field(data, url.rstrip("/").split("/")[-1].split("?")[0] or "archivo.fits", {"archive": "URL directa", "download_url": url})
        except Exception as exc:
            st.error(str(exc))

# ================================================================ 2. IMAGEN Y REGION
raw = st.session_state.get("field_bytes")
name = st.session_state.get("field_name")
source = st.session_state.get("field_source") or {}
if not raw:
    st.info("Elige un archivo arriba para ver la imagen.")
    st.stop()

st.markdown("### 2. Imagen y región de estudio")
st.success("Cargado: %s · %.1f MB · origen: %s" % (name, len(raw) / 1048576, source.get("archive", "—")))

hdu_index = None
if str(name).lower().endswith((".fits", ".fit", ".fits.gz")):
    try:
        hkey = (name, len(raw))
        if st.session_state.get("hdus_key") != hkey:
            st.session_state["hdus"] = list_image_hdus(raw)
            st.session_state["hdus_key"] = hkey
        hdus = st.session_state["hdus"]
    except Exception as exc:
        st.error("No se pudo leer el FITS: %s" % exc)
        st.stop()
    if not hdus:
        st.error("Este FITS no contiene ninguna extensión de imagen 2D (puede ser una tabla o un cubo). Elige otro archivo.")
        st.stop()
    default = next((i for i, h in enumerate(hdus) if h["name"] == "SCI"), 0)
    choice = st.selectbox("Extensión", range(len(hdus)), index=default,
                          format_func=lambda i: "%s[%d] %s" % (hdus[i]["name"], hdus[i]["index"], "x".join(map(str, hdus[i]["shape"]))))
    hdu_index = hdus[choice]["index"]
    if hdus[choice]["name"] != "SCI" and any(h["name"] == "SCI" for h in hdus):
        st.warning("Estás analizando %s, no SCI. ERR/WHT/VAR no son mapas del cielo." % hdus[choice]["name"])

pkey = (name, len(raw), hdu_index)
if st.session_state.get("parsed_key") != pkey:
    try:
        st.session_state["parsed"] = parse_field(raw, name, hdu_index)
        st.session_state["parsed_key"] = pkey
        st.session_state.pop("results", None)
    except Exception as exc:
        st.exception(exc)
        st.stop()
data, meta = st.session_state["parsed"]

bbox = finite_bbox(data)
if bbox is None:
    st.error("La extensión no tiene píxeles válidos (todo NaN).")
    st.stop()
r0, r1, c0, c1 = bbox
H, W = data.shape
max_side = min(H, W, MAX_ANALYSIS_SIDE)
def_side = min(r1 - r0, c1 - c0, 1024)
if max_side < 32:
    st.error("La imagen es demasiado pequeña (%dx%d) para analizarla." % (W, H))
    st.stop()


def _slider(col, label, lo, hi, default):
    """Slider tolerante: si el rango colapsa (lo == hi) no hay nada que elegir."""
    if hi <= lo:
        col.caption("%s: %d (fijo)" % (label, lo))
        return lo
    return col.slider(label, lo, hi, int(np.clip(default, lo, hi)))


s1, s2, s3 = st.columns(3)
side = _slider(s1, "Tamaño de la región (px)", 32, max_side, min(def_side, max_side))
y0 = _slider(s2, "Fila inicial (y)", 0, H - side, (r0 + r1 - side) // 2)
x0 = _slider(s3, "Columna inicial (x)", 0, W - side, (c0 + c1 - side) // 2)

crop_raw = data[y0:y0 + side, x0:x0 + side]
nan_frac = float(np.mean(~np.isfinite(crop_raw)))
crop, (p_lo, p_hi) = stretch(crop_raw)

left, right = st.columns([3, 2])
left.image(preview_with_box(data, y0, x0, side), caption="Imagen completa (%dx%d). Recuadro = región de estudio." % (W, H), use_container_width=True)
right.image(crop, caption="Región: x=%d–%d, y=%d–%d" % (x0, x0 + side, y0, y0 + side), use_container_width=True, clamp=True)
right.markdown(
    "<div class='cms-prov'>Telescopio / instrumento: %s / %s<br>Filtro: %s<br>Objeto (cabecera): %s<br>"
    "Fecha (cabecera): %s<br>Programa (cabecera): %s<br>Extensión: %s<br>Unidades: %s</div>"
    % (meta.get("telescope") or "—", meta.get("instrument") or "—", meta.get("filter") or "—", meta.get("target_header") or "—",
       meta.get("date_obs") or "—", meta.get("program") or "—", meta.get("hdu") or "—", meta.get("bunit") or "—"),
    unsafe_allow_html=True)
if nan_frac > 0.05:
    st.warning("%.0f%% de la región son píxeles vacíos (borde del mosaico). Se rellenan con la mediana, lo que crea "
               "bordes artificiales: mueve o reduce la región." % (100 * nan_frac))
if not meta.get("is_fits"):
    st.warning("Imagen no científica (%s): el estiramiento y la compresión alteran la estadística. Usa FITS para conclusiones." % meta.get("format"))

# ================================================================ 3. SONIFICACION
st.markdown("### 3. Escuchar la región")
st.caption("Sonificación, no sonido real: el espacio casi no transmite sonido. La región se recorre de izquierda a "
           "derecha en 8 s; el tono sigue el brillo medio de cada columna y el volumen su contraste. Compara la región "
           "real con un subrogado IAAFT, que tiene el mismo espectro e histograma pero la estructura espacial barajada: "
           "lo que oigas distinto es lo que el nulo intenta detectar.")
if st.button("Generar audio"):
    from scoring import downsample_for_null, iaaft_surrogate
    small = downsample_for_null(crop, max_side=256)
    surr = iaaft_surrogate(small, np.random.default_rng(SEED))
    st.session_state["audio"] = (sonify(small, ref=surr), sonify(surr, ref=small), pkey + (x0, y0, side))
aud = st.session_state.get("audio")
if aud and aud[2] == pkey + (x0, y0, side):
    a1, a2 = st.columns(2)
    a1.markdown("Región real")
    a1.audio(aud[0], sample_rate=22050)
    a2.markdown("Subrogado IAAFT (ruido con el mismo espectro)")
    a2.audio(aud[1], sample_rate=22050)

# ================================================================ 4. ANALISIS
st.markdown("### 4. Analizar la región")
if st.button("Analizar", type="primary"):
    try:
        from candidate_export import build_candidate, dumps_candidate
        from materials_map import interpret
        from nos_morphological import morphological_nos
        from provenance import assess_provenance
        from scoring import cheap_descriptor_pvalues, cheap_score_from_image, fdr_decision, surrogate_null_test

        meta_full = dict(meta)
        meta_full.update({"crop": {"x0": int(x0), "y0": int(y0), "side": int(side)}, "nan_fraction": round(nan_frac, 4),
                          "normalization": {"method": "percentil 2-98 sobre pixeles finitos, NaN -> mediana, recorte [0,1]",
                                            "p2": p_lo, "p98": p_hi},
                          "null": {"method": "iaaft", "n_simulations": int(n_null), "seed": SEED},
                          "source": source, "active_descriptors": active})
        src = source.get("download_url") or source.get("uri") or source.get("archive", "")
        prov = assess_provenance(str(name), meta_full, src)
        with st.spinner("Calculando descriptores y %d subrogados…" % n_null):
            results = run_plugins(crop, active)
            mc = surrogate_null_test(crop, None, cheap_score_from_image, n_simulations=n_null, seed=SEED)
            fdr = fdr_decision(cheap_descriptor_pvalues(crop, n_simulations=n_null, seed=SEED + 1), n_simulations=n_null)
        mc["fdr"] = fdr
        materials = interpret(results, float(mc.get("z_score") or 0), float(mc.get("p_value") or 1),
                              fdr_pass=bool(fdr.get("fdr_pass")), provenance=prov, metadata=meta_full)
        nos = morphological_nos(results)
        cand = build_candidate(name, results, materials, prov, nos, mc, meta_full, src)
        st.session_state["results"] = {"key": pkey + (x0, y0, side), "results": results, "mc": mc, "fdr": fdr,
                                       "materials": materials, "nos": nos, "prov": prov, "json": dumps_candidate(cand)}
    except Exception as exc:
        st.exception(exc)

R = st.session_state.get("results")
if R and R["key"] == pkey + (x0, y0, side):
    mat, fdr, mc, nos, prov = R["materials"], R["fdr"], R["mc"], R["nos"], R["prov"]
    st.markdown("**%s** — %s" % (mat.get("state"), mat.get("verdict")))
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Familia dominante", mat.get("dominant_label") or "—")
    k2.metric("FDR", "pasa" if fdr.get("fdr_pass") else "no pasa")
    k3.metric("p (score global)", "%.3f" % mc.get("p_value", 1.0))
    k4.metric("Procedencia", "%s (%.2f)" % (prov.get("verdict"), float(prov.get("trust_score") or 0)))
    if fdr.get("can_pass") is False:
        st.warning("Con %d subrogados el FDR no puede pasar: sube el número en ajustes avanzados." % n_null)
    st.dataframe([{"descriptor": k, "p": round(v, 4), "sobrevive FDR": "sí" if k in fdr.get("passed_descriptors", []) else "no"}
                  for k, v in fdr.get("p_values", {}).items()], hide_index=True, use_container_width=True)
    if mat.get("instrument_warning_reason"):
        st.warning("Aviso de instrumento: %s" % mat["instrument_warning_reason"])
    fam = mat.get("family_scores") or {}
    if any(fam.values()):
        st.bar_chart(fam)
        st.caption("Cobertura de descriptores por familia: %s" % ", ".join("%s %.0f%%" % (k, 100 * v) for k, v in (mat.get("family_coverage") or {}).items()))
    errors = {k: v["error"] for k, v in R["results"].items() if isinstance(v, dict) and "error" in v}
    if errors:
        st.error("Descriptores que fallaron (excluidos, no imputados): " + "; ".join("%s: %s" % kv for kv in errors.items()))
    st.caption("Análogo de laboratorio (descriptivo, no identificación): %s. Seguimiento: %s." % (mat.get("lab_analog"), mat.get("followup")))
    st.download_button("Descargar informe JSON", R["json"], file_name="cms80_%s_x%d_y%d_s%d.json" % (str(name).split(".")[0], x0, y0, side))

with st.expander("Cómo citar y límites"):
    st.markdown(MAST_ACK)
    st.markdown("- Un PNG de prensa no es un dato científico: usa FITS calibrados (i2d, drz, drc, cal, flt).\n"
                "- El nulo dice si hay estructura que el ruido con el mismo espectro e histograma no explica; no dice si el objeto es raro en astronomía.\n"
                "- La familia y el análogo son descripciones morfológicas, no identificaciones de material.")
