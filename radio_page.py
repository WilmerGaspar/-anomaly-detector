"""Página 'Señales de radio': biblioteca de firmas, carga de datos y detectores con nulo."""
from __future__ import annotations

import csv
import io

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import radio_signals as rs

GREEN, DIM, ACCENT = "#33ff66", "#2a5a35", "#ffb347"
# Memoria de Streamlit Cloud (~690 MB garantizados). Medido fuera de la app: periodicidad con
# 4 M muestras, pico 318 MB; con 16 M, 966 MB (la app se caeria).
MAX_RADIO_VALUES = 4_000_000
MAX_RADIO_MB = 64


def _layout(fig, height=300, **kw):
    fig.update_layout(template="plotly_dark", paper_bgcolor="#020803", plot_bgcolor="#020803", height=height,
                      font=dict(color="#b7ffc2", size=12), margin=dict(l=50, r=20, t=40, b=40), **kw)
    return fig


def _verdict(res):
    if res.get("detected") and res.get("likely_rfi"):
        return "🟡", "Detectado, probable interferencia terrestre"
    if res.get("detected"):
        return "🟢", "Firma detectada por encima de su nulo"
    return "⚪", "Sin detección (compatible con ruido)"


def _load_array(up):
    name = up.name.lower()
    if up.size > MAX_RADIO_MB * 1048576:
        raise ValueError("pesa %.0f MB y el máximo es %d MB. Recórtalo o promédialo (binning) antes de cargarlo."
                         % (up.size / 1048576, MAX_RADIO_MB))
    raw = up.getvalue()
    if name.endswith(".npy"):
        return np.load(io.BytesIO(raw), allow_pickle=False)
    if name.endswith((".fits", ".fit")):
        from astropy.io import fits
        with fits.open(io.BytesIO(raw), memmap=False) as hdul:
            for h in hdul:
                if getattr(h, "is_image", False) and h.data is not None:
                    return np.asarray(h.data, dtype=float).squeeze()
        raise ValueError("El FITS no tiene datos de imagen.")
    text = raw.decode("utf-8", errors="replace")
    rows = [r for r in csv.reader(io.StringIO(text)) if r and not r[0].lstrip().startswith("#")]
    vals = []
    for r in rows:
        try:
            vals.append([float(v) for v in r])
        except ValueError:
            continue                                  # cabecera
    return np.array(vals, dtype=float).squeeze()


def _library():
    st.markdown("#### Biblioteca de firmas")
    st.caption("Cada firma se genera con su ecuación física (sin datos memorizados). Úsalas para ver cómo responde "
               "cada detector antes de cargar tus datos.")
    cols = st.columns(len(rs.LIBRARY))
    for col, (key, info) in zip(cols, rs.LIBRARY.items()):
        col.markdown("**%s**" % info["name"])
        col.caption("%s · %s" % (info["data"], info["physics"]))
        if col.button("Generar ejemplo", key="gen_" + key):
            rng_seed = int(np.random.default_rng().integers(1_000_000))
            if key == "pulsar":
                st.session_state["radio"] = {"kind": "series", "data": rs.gen_pulsar(2 ** 15, 1e-3, 0.0893, amp=0.12, seed=rng_seed),
                                             "dt": 1e-3, "label": "Ejemplo: púlsar P = 0.0893 s"}
            elif key == "dispersed_burst":
                dyn, f = rs.gen_dispersed_burst(dm=500, amp=0.5, seed=rng_seed)
                st.session_state["radio"] = {"kind": "dyn_ft", "data": dyn, "freqs_ghz": f, "dt": 1e-3,
                                             "label": "Ejemplo: ráfaga con DM = 500 pc/cm³"}
            elif key == "drifting_tone":
                st.session_state["radio"] = {"kind": "dyn_tf", "data": rs.gen_drifting_tone(drift=-0.15, amp=0.6, seed=rng_seed),
                                             "df_hz": 3.0, "dt": 10.0, "label": "Ejemplo: portadora con deriva −0.15 Hz/s"}
            elif key == "rfi_impulse":
                dyn = rs.gen_rfi_impulse(seed=rng_seed)
                st.session_state["radio"] = {"kind": "dyn_ft", "data": dyn, "freqs_ghz": rs.gen_dispersed_burst()[1],
                                             "dt": 1e-3, "label": "Ejemplo: impulso RFI sin dispersión"}
            elif key == "rfi_tone":
                st.session_state["radio"] = {"kind": "dyn_tf", "data": rs.gen_drifting_tone(drift=0.0, amp=0.8, seed=rng_seed),
                                             "df_hz": 3.0, "dt": 10.0, "label": "Ejemplo: portadora fija (RFI)"}
            st.session_state.pop("radio_result", None)


def _upload():
    st.markdown("#### Tus datos")
    kind = st.radio("Tipo de dato", ["Serie temporal (1 columna)", "Espectro dinámico: canales × tiempo (ráfagas)",
                                     "Espectrograma: tiempo × canales finos (portadoras)"], key="radio_kind")
    up = st.file_uploader("Archivo (.csv, .txt, .npy o .fits)", type=["csv", "txt", "npy", "fits", "fit"], key="radio_up")
    c1, c2, c3 = st.columns(3)
    if kind.startswith("Serie"):
        dt = c1.number_input("Intervalo de muestreo dt (s)", min_value=1e-9, value=1e-3, format="%.6g")
    elif kind.startswith("Espectro dinámico"):
        dt = c1.number_input("dt por muestra (s)", min_value=1e-9, value=1e-3, format="%.6g")
        f_top = c2.number_input("Frecuencia del primer canal (MHz)", value=1500.0)
        f_bot = c3.number_input("Frecuencia del último canal (MHz)", value=1200.0)
    else:
        dt = c1.number_input("dt por espectro (s)", min_value=1e-6, value=10.0, format="%.6g")
        df = c2.number_input("Ancho de canal (Hz)", min_value=1e-6, value=3.0, format="%.6g")
    if up is not None and st.button("Usar este archivo", type="primary"):
        try:
            arr = _load_array(up)
            if arr.size > MAX_RADIO_VALUES:
                raise ValueError("tiene %.1f millones de valores y el máximo es %.0f millones (memoria del servidor). "
                                 "Recórtalo o promédialo (binning) antes de cargarlo." % (arr.size / 1e6, MAX_RADIO_VALUES / 1e6))
            if kind.startswith("Serie"):
                arr = arr[:, -1] if arr.ndim == 2 else arr
                st.session_state["radio"] = {"kind": "series", "data": arr.astype(float), "dt": dt, "label": up.name}
            elif arr.ndim != 2:
                raise ValueError("Se esperaba una matriz 2D y llegó una de %d dimensiones." % arr.ndim)
            elif kind.startswith("Espectro dinámico"):
                f = np.linspace(f_top, f_bot, arr.shape[0]) / 1000.0
                st.session_state["radio"] = {"kind": "dyn_ft", "data": arr, "freqs_ghz": f, "dt": dt, "label": up.name}
            else:
                st.session_state["radio"] = {"kind": "dyn_tf", "data": arr, "df_hz": df, "dt": dt, "label": up.name}
            st.session_state.pop("radio_result", None)
        except Exception as exc:
            st.error("No se pudo leer el archivo: %s" % exc)
    st.caption("Fuentes públicas (descarga allí; esta app no se conecta a ellas): " +
               " · ".join("[%s](%s)" % (n, u) for n, u, _ in rs.PUBLIC_CATALOGS) +
               ". Formatos de radiotelescopio (filterbank .fil, HDF5 .h5) hay que convertirlos antes a .npy o .fits. "
               "Máximo %d MB y %.0f millones de valores." % (MAX_RADIO_MB, MAX_RADIO_VALUES / 1e6))


def _show(res, data):
    icon, text = _verdict(res)
    st.markdown("### %s %s" % (icon, text))
    st.caption(res.get("note") or "")
    st.caption("Método: " + res.get("method", ""))


def render_radio_page():
    st.markdown("### 📡 Señales de radio")
    st.caption("Busca tres firmas físicas con su propio nulo: periodicidad (púlsar), dispersión f⁻² (ráfagas tipo "
               "FRB) y deriva Doppler (portadoras de banda estrecha). Una detección significa 'por encima de su "
               "nulo', no 'origen artificial' ni 'origen astrofísico': DM ≈ 0 y deriva ≈ 0 se marcan como RFI.")
    _library()
    _upload()
    D = st.session_state.get("radio")
    if not D:
        st.info("Genera un ejemplo de la biblioteca o carga un archivo.")
        return
    st.markdown("---")
    st.markdown("**Datos:** %s" % D["label"])
    arr = np.asarray(D["data"], dtype=float)
    if D["kind"] == "series":
        fig = go.Figure(go.Scatter(y=arr[: min(len(arr), 5000)], mode="lines", line=dict(color=DIM)))
        st.plotly_chart(_layout(fig, 200, title="Serie (primeras 5000 muestras)"), use_container_width=True)
    else:
        fig = go.Figure(go.Heatmap(z=arr, colorscale="Greys", showscale=False))
        st.plotly_chart(_layout(fig, 280, title="Espectro dinámico (%d × %d)" % arr.shape), use_container_width=True)

    if st.button("Buscar firmas", type="primary"):
        with st.spinner("Buscando con nulos…"):
            try:
                if D["kind"] == "series":
                    st.session_state["radio_result"] = ("period", rs.periodicity_search(arr, D["dt"]))
                elif D["kind"] == "dyn_ft":
                    st.session_state["radio_result"] = ("disp", rs.dispersion_search(arr, D["freqs_ghz"], D["dt"]))
                else:
                    st.session_state["radio_result"] = ("drift", rs.drift_search(arr, D["df_hz"], D["dt"]))
            except Exception as exc:
                st.exception(exc)
    RR = st.session_state.get("radio_result")
    if not RR:
        return
    kind, res = RR
    _show(res, D)
    if kind == "period":
        if "best_period_s" in res:
            c1, c2, c3 = st.columns(3)
            c1.metric("Periodo", "%.6g s" % res["best_period_s"],
                      "± %.2g s" % res["period_err_s"] if res.get("period_err_s") else None, delta_color="off")
            c2.metric("Armónicos sumados", res["harmonics_summed"])
            c3.metric("Prob. falsa alarma", "%.2g" % res["fap"])
            a, b = st.columns(2)
            fig = go.Figure(go.Scatter(x=res["spectrum_f"], y=res["spectrum_p"], mode="lines", line=dict(color=DIM)))
            fig.add_vline(x=res["best_freq_hz"], line_color=ACCENT)
            a.plotly_chart(_layout(fig, title="Espectro normalizado (ruido ≈ 1)", xaxis_title="Hz"), use_container_width=True)
            fig = go.Figure(go.Scatter(y=res["folded_profile"] * 2, mode="lines+markers", line=dict(color=ACCENT)))
            b.plotly_chart(_layout(fig, title="Perfil plegado (2 vueltas)", xaxis_title="fase × 64"), use_container_width=True)
        st.caption("Periodo máximo buscado: %.4g s (al menos 20 ciclos en la serie)." % res.get("max_period_s", float("nan")))
    elif kind == "disp":
        c1, c2, c3 = st.columns(3)
        c1.metric("DM", "%.1f pc/cm³" % res["dm"])
        c2.metric("S/N", "%.1f" % res["snr"])
        c3.metric("Prob. falsa alarma", "%.2g" % res["fap"])
        a, b = st.columns(2)
        fig = go.Figure(go.Scatter(x=res["dm_grid"], y=res["dm_curve"], mode="lines", line=dict(color=ACCENT)))
        a.plotly_chart(_layout(fig, title="S/N frente a DM", xaxis_title="DM (pc/cm³)"), use_container_width=True)
        fig = go.Figure(go.Scatter(y=res["dedispersed"], mode="lines", line=dict(color=DIM)))
        b.plotly_chart(_layout(fig, title="Serie desdispersada al mejor DM"), use_container_width=True)
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Deriva", "%.4f Hz/s" % res["drift_hz_s"])
        c2.metric("S/N", "%.1f" % res["snr"])
        c3.metric("Prob. falsa alarma", "%.2g" % res["fap"])
    st.caption("Control empírico: S/N observado %.1f frente al máximo de %d nulos permutados %.1f."
               % (res.get("snr", float("nan")), len(res.get("null_snr", [])), max(res.get("null_snr", [float("nan")])))
               if res.get("null_snr") else "")

    st.markdown("#### Comparar con un catálogo público")
    cat = st.file_uploader("CSV exportado del catálogo (columnas P0 en s y/o DM)", type=["csv"], key="radio_cat")
    if cat is not None:
        rows = list(csv.DictReader(io.StringIO(cat.getvalue().decode("utf-8", errors="replace"))))
        m = rs.crossmatch_catalog(rows, period_s=res.get("best_period_s"), dm=res.get("dm"),
                                  period_err_s=res.get("period_err_s"))
        tol = max(0.002, 3 * res["period_err_s"] / res["best_period_s"]) if res.get("period_err_s") else 0.002
        if m:
            st.success("%d coincidencias en el catálogo (P0 ±%.2g %% incluyendo armónicos, DM ±10 %%): lo más probable "
                       "es que sea un objeto conocido." % (len(m), 100 * tol))
            st.dataframe([dict(mm["row"], armonico=mm["harmonic"]) for mm in m[:50]], use_container_width=True)
        else:
            st.info("Sin coincidencias en %d filas del catálogo. Revisa unidades (P0 en segundos) antes de concluir." % len(rows))
