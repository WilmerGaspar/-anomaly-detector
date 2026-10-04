"""Panel analitico de resultados (paso 4 de app.py).

Cada pestana responde a una pregunta concreta y muestra el nulo junto al
valor observado, para que ningun numero aparezca sin su referencia.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from analytics import bh_curve, flatten_numeric, surrogate_preview

GREEN, DIM, ACCENT, GRID = "#33ff66", "#2a5a35", "#ffb347", "#0d2614"
DESCRIPTOR_HELP = {
    "aniso": "Coef. de variación de la energía del gradiente: alto si la estructura se concentra en bordes.",
    "flatness_lag1": "Curtosis de incrementos a 1 px: > 3 indica saltos raros pero intensos (bordes, filamentos finos).",
    "flatness_lag4": "Curtosis de incrementos a 4 px: intermitencia a escala intermedia.",
    "incr_skew": "Asimetría de incrementos: subidas y bajadas de brillo no equivalentes (frentes, choques).",
}


def _layout(fig, height=340, **kw):
    fig.update_layout(template="plotly_dark", paper_bgcolor="#020803", plot_bgcolor="#020803", height=height,
                      font=dict(color="#b7ffc2", size=12), margin=dict(l=50, r=20, t=40, b=40),
                      legend=dict(orientation="h", y=-0.2), **kw)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def _null_hist(fig, null, observed, row=None, col=None, name=""):
    kw = dict(row=row, col=col) if row else {}
    fig.add_trace(go.Histogram(x=null, nbinsx=30, marker_color=DIM, name="nulo IAAFT", showlegend=row in (None, 1) and col in (None, 1)), **kw)
    fig.add_vline(x=observed, line_color=ACCENT, line_width=2, **kw)
    q05, q95 = np.percentile(null, [5, 95]) if len(null) else (np.nan, np.nan)
    if np.isfinite(q05):
        fig.add_vrect(x0=q05, x1=q95, fillcolor=GREEN, opacity=0.07, line_width=0, **kw)


# ---------------------------------------------------------------- pestañas

def _tab_null(R, n_null):
    fdr, details, mc = R["fdr"], R["details"], R["mc"]
    alpha = fdr.get("alpha", 0.05)
    bh = {r["descriptor"]: r["threshold"] for r in bh_curve(fdr.get("p_values", {}), alpha)}
    rows = []
    for k, d in details.items():
        rows.append({"descriptor": k, "observado": d["observed"], "media nulo": d["null_mean"],
                     "nulo 5–95 %": "%.3f – %.3f" % (d["null_q05"], d["null_q95"]), "z": d["z"], "p": d["p"],
                     "umbral BH": bh.get(k, np.nan), "sobrevive FDR": "sí" if k in fdr.get("passed_descriptors", []) else "no",
                     "qué mide": DESCRIPTOR_HELP.get(k, "")})
    st.dataframe(pd.DataFrame(rows).style.format({"observado": "{:.4f}", "media nulo": "{:.4f}", "z": "{:+.2f}", "p": "{:.4f}",
                                                    "umbral BH": "{:.4f}"}),
                 hide_index=True, use_container_width=True)
    st.caption("Test de una cola: solo cuenta que el observado esté POR ENCIMA del nulo (z positivo). "
               "p = (1 + #nulo ≥ obs)/(n + 1); p mínimo posible con %d subrogados = %.4f." % (n_null, 1 / (n_null + 1)))

    beta = R["analytics"]["spectrum"]["fit"]["beta"]
    if np.isfinite(beta) and beta > 2.5:
        st.info("β = %.2f: con espectros empinados, los subrogados IAAFT tienen más curtosis que el campo real "
                "(medido en campos gaussianos: 3.24 con β=2.8 y 4.38 con β=3.5, frente a 3.0). El test sigue sin dar "
                "falsos positivos, pero `aniso` y `flatness_lag1` pierden sensibilidad: un \"no pasa\" aquí no descarta "
                "estructura fina. Mira también las pestañas Escalas y Familias (crestas)." % beta)
    keys = list(details)
    fig = make_subplots(rows=2, cols=2, subplot_titles=["%s  (z=%+.2f, p=%.3f)" % (k, details[k]["z"], details[k]["p"]) for k in keys])
    for i, k in enumerate(keys[:4]):
        _null_hist(fig, details[k]["null"], details[k]["observed"], row=i // 2 + 1, col=i % 2 + 1)
    st.plotly_chart(_layout(fig, height=520, title="Distribución nula de cada descriptor (línea naranja = región, banda = 5–95 % del nulo)"),
                    use_container_width=True)

    c1, c2 = st.columns(2)
    curve = bh_curve(fdr.get("p_values", {}), alpha)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["rank"] for r in curve], y=[r["p"] for r in curve], mode="markers+text", text=[r["descriptor"] for r in curve],
                             textposition="top center", marker=dict(color=ACCENT, size=10), name="p ordenados"))
    fig.add_trace(go.Scatter(x=[r["rank"] for r in curve], y=[r["threshold"] for r in curve], mode="lines", line=dict(color=GREEN, dash="dash"),
                             name="umbral BH (i/m·α)"))
    fig.add_trace(go.Scatter(x=[1, max(len(curve), 1)], y=[1 / (n_null + 1)] * 2, mode="lines", line=dict(color=DIM, dash="dot"),
                             name="p mínimo posible"))
    fig.update_yaxes(type="log", title="p")
    fig.update_xaxes(title="rango", dtick=1)
    c1.plotly_chart(_layout(fig, title="Benjamini-Hochberg (α=%.2f)" % alpha), use_container_width=True)

    null = mc.get("null_scores") or []
    if null:
        fig = go.Figure()
        _null_hist(fig, null, mc.get("observed_score", np.nan))
        fig.update_xaxes(title="score global")
        c2.plotly_chart(_layout(fig, title="Score global (z=%+.2f, p=%.3f)"
                                % (mc.get("z_score", 0), mc.get("p_value", 1))), use_container_width=True)
    c1.caption("Pasan los descriptores cuyo p queda bajo la línea discontinua (y todos los de rango menor).")
    c2.caption("Región (línea naranja) frente a %d subrogados. Análisis del nulo a %s px de lado (región reducida)." % (len(null), mc.get("analysis_side_px", "—")))


def _tab_spectrum(A):
    sp = A["spectrum"]
    fit = sp["fit"]
    k, p, ps = np.array(sp["k"]), np.array(sp["p"]), np.array(sp["p_surrogate"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k, y=p, mode="lines+markers", marker=dict(size=4), line=dict(color=ACCENT), name="región"))
    fig.add_trace(go.Scatter(x=k, y=ps, mode="lines", line=dict(color=DIM), name="subrogado IAAFT"))
    if np.isfinite(fit["beta"]):
        kk = np.linspace(fit["k_min"], fit["k_max"], 20)
        fig.add_trace(go.Scatter(x=kk, y=10 ** (fit["intercept"]) * kk ** (-fit["beta"]), mode="lines",
                                 line=dict(color=GREEN, dash="dash"), name="ajuste k^-β"))
    fig.update_xaxes(type="log", title="k (ciclos/píxel)")
    fig.update_yaxes(type="log", title="P(k)")
    c1, c2 = st.columns([3, 1])
    c1.plotly_chart(_layout(fig, height=420, title="Espectro de potencia radial"), use_container_width=True)
    c2.metric("β (pendiente)", "%.2f" % fit["beta"] if np.isfinite(fit["beta"]) else "—")
    c2.metric("R² del ajuste", "%.3f" % fit["r2"] if np.isfinite(fit["r2"]) else "—")
    c2.caption("Rango del ajuste: k = %.2f–%.2f (%d puntos)." % (fit["k_min"], fit["k_max"], fit["n_points"]))
    c2.caption("Referencias: β≈11/3 turbulencia de Kolmogorov (densidad), β≈2 bordes/escalones, β≈0 ruido blanco. "
               "Un R² bajo indica que no hay ley de potencia única (dos regímenes o un pico).")
    c2.caption("El subrogado tiene el mismo espectro por construcción: las curvas deben solaparse. "
               "Si no lo hacen, el nulo no es válido para esta región.")


def _tab_scales(A):
    sc, inc = A["scales"], A["increments"]
    lags = sc["lags"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lags + lags[::-1], y=sc["flatness_null_q95"] + sc["flatness_null_q05"][::-1], fill="toself",
                             fillcolor="rgba(51,255,102,0.12)", mode="lines", line=dict(width=0), name="nulo 5–95 %", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=lags, y=sc["flatness_null_mean"], mode="lines", line=dict(color=DIM, dash="dash"), name="media nulo"))
    fig.add_trace(go.Scatter(x=lags, y=sc["flatness"], mode="lines+markers", line=dict(color=ACCENT), name="región"))
    fig.add_hline(y=3, line_color=GREEN, line_dash="dot", annotation_text="gaussiano (3)")
    fig.update_xaxes(type="log", title="escala (px)", tickvals=lags, ticktext=[str(l) for l in lags])
    fig.update_yaxes(title="curtosis de incrementos")
    c1, c2 = st.columns(2)
    c1.plotly_chart(_layout(fig, title="Intermitencia por escala (%d subrogados)" % sc["n_surrogates"]), use_container_width=True)
    c1.caption("Si la curva naranja sale por encima de la banda a escalas pequeñas, hay estructura fina "
               "(filamentos, bordes) que el ruido equivalente no tiene.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=inc["x"], y=inc["observed"], mode="lines", line=dict(color=ACCENT), name="región"))
    fig.add_trace(go.Scatter(x=inc["x"], y=inc["surrogate"], mode="lines", line=dict(color=DIM), name="subrogado IAAFT"))
    fig.add_trace(go.Scatter(x=inc["x"], y=inc["gaussian"], mode="lines", line=dict(color=GREEN, dash="dot"), name="gaussiana"))
    fig.update_yaxes(type="log", title="densidad", range=[-5, 0])
    fig.update_xaxes(title="incremento / σ (lag %d px)" % inc["lag"])
    c2.plotly_chart(_layout(fig, title="Distribución de incrementos"), use_container_width=True)
    c2.caption("Colas más pesadas que el subrogado = saltos de brillo bruscos más frecuentes de lo que explica el espectro.")


def _tab_local(A, crop):
    lm = A["local_map"]
    z = np.array(lm["z"], dtype=float)
    if np.all(~np.isfinite(z)):
        st.info("Región demasiado pequeña para el mapa local: cada tesela necesita al menos 16 px de lado.")
        return
    g = lm["grid"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Región", "z local de %s" % lm["metric"]], horizontal_spacing=0.08)
    # Solo para mostrar: como mucho 256 px de lado (media por bloques). Con la region completa
    # (1024 px = 1 M valores, 2048 px = 4 M) el grafico ocupaba decenas de MB por sesion y,
    # analisis tras analisis, llevaba a Streamlit Cloud al limite de memoria.
    show = np.asarray(crop, dtype=np.float32)
    b = int(np.ceil(max(show.shape) / 256))
    if b > 1:
        h, w = (show.shape[0] // b) * b, (show.shape[1] // b) * b
        show = show[:h, :w].reshape(h // b, b, w // b, b).mean(axis=(1, 3))
    fig.add_trace(go.Heatmap(z=show[::-1], colorscale="Greys", reversescale=True, showscale=False), row=1, col=1)
    lim = float(np.clip(np.nanmax(np.abs(z)), 3.0, 10.0))     # escala de color saturada en |z| = 10
    fig.add_trace(go.Heatmap(z=z[::-1], zmin=-lim, zmax=lim, colorscale="RdBu", reversescale=True,
                             text=np.round(z[::-1], 1), texttemplate="%{text}", colorbar=dict(title="z")), row=1, col=2)
    for c in (1, 2):
        fig.update_xaxes(showticklabels=False, row=1, col=c)
        fig.update_yaxes(showticklabels=False, scaleanchor="x" if c == 1 else "x2", row=1, col=c)
    st.plotly_chart(_layout(fig, height=440), use_container_width=True)
    st.caption("Región dividida en %d×%d teselas de %s px; cada una se compara con %d subrogados IAAFT propios. "
               "Rojo = más intermitente que su nulo (z > 2 merece mirarse; el color satura en |z| = 10). Exploratorio: sin corrección por "
               "comparaciones múltiples y con pocos subrogados; no sustituye al FDR global. Los IAAFT tienden a "
               "tener curtosis algo mayor que un campo gaussiano, así que z negativos son esperables y no significan nada."
               % (g, g, "×".join(map(str, lm.get("tile_px", []))), lm.get("n_surrogates", 0)))


def _tab_families(R):
    mat = R["materials"]
    fam = mat.get("family_scores") or {}
    cov = mat.get("family_coverage") or {}
    if not any(fam.values()):
        st.info("Sin puntuaciones de familia (activa más descriptores).")
        return
    order = sorted(fam, key=fam.get, reverse=True)
    fig = go.Figure(go.Bar(x=[fam[k] for k in order], y=order, orientation="h",
                           marker_color=[ACCENT if k == order[0] else DIM for k in order],
                           text=["cobertura %.0f%%" % (100 * cov.get(k, 0)) for k in order], textposition="auto"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="puntuación (ponderada por cobertura)")
    st.plotly_chart(_layout(fig, height=320, title="Familias morfológicas"), use_container_width=True)
    if len(order) > 1:
        st.caption("Margen entre la primera y la segunda familia: %.3f. Un margen pequeño significa clasificación ambigua."
                   % (fam[order[0]] - fam[order[1]]))
    rid = R["results"].get("ridges") or {}
    per = R["results"].get("periodicity") or {}
    if rid.get("n_spike_components"):
        st.caption("Crestas descartadas como picos de difracción (salen radialmente de una fuente puntual mucho más "
                   "brillante): %d componentes, %.2f %% de la región." % (rid["n_spike_components"], 100 * rid.get("spike_fraction", 0)))
    if "filament_excess" in rid:
        st.caption("Crestas largas: %.2f %% de la región frente a %.2f %% en subrogados IAAFT (exceso %.4f; en controles "
                   "sintéticos el ruido da ≤ 0.001 y los filamentos 0.011–0.030). Rayas en la FFT (líneas rectas): %d."
                   % (100 * rid["filament_fraction"], 100 * rid["filament_fraction_null"], rid["filament_excess"], per.get("n_streaks", 0)))
    st.caption("Análogo de laboratorio (descriptivo, no identificación): %s. Seguimiento: %s." % (mat.get("lab_analog"), mat.get("followup")))
    nos = R.get("nos")
    if isinstance(nos, dict) and nos:
        with st.expander("NOS morfológico (orientativo, no es evidencia)"):
            warn = (nos.get("empirical") or {}).get("warning")
            if warn:
                st.warning(warn)
            st.caption("v0: distancia a 5 prototipos escritos a mano. No interviene en el semáforo ni en el estado.")
            st.json(nos)


def _tab_descriptors(R):
    rows = flatten_numeric(R["results"])
    errors = {k: v["error"] for k, v in R["results"].items() if isinstance(v, dict) and "error" in v}
    if errors:
        st.error("Descriptores que fallaron (excluidos, no imputados): " + "; ".join("%s: %s" % kv for kv in errors.items()))
    if not rows:
        st.info("Sin descriptores numéricos.")
        return None
    df = pd.DataFrame(rows)
    plugin = st.selectbox("Filtrar por plugin", ["todos"] + sorted(df["plugin"].unique()), key="desc_filter")
    view = df if plugin == "todos" else df[df["plugin"] == plugin]
    st.dataframe(view, hide_index=True, use_container_width=True, height=380)
    st.caption("%d valores numéricos de %d plugins." % (len(df), df["plugin"].nunique()))
    return df


def _tab_compare(crop):
    small, surr = surrogate_preview(crop, seed=80)
    c1, c2, c3 = st.columns(3)
    c1.image(np.clip(small, 0, 1), caption="Región (reducida)", use_container_width=True, clamp=True)
    c2.image(np.clip(surr, 0, 1), caption="Subrogado IAAFT: mismo espectro e histograma", use_container_width=True, clamp=True)
    diff = np.abs(np.gradient(small)[0]) + np.abs(np.gradient(small)[1])
    c3.image(diff / (diff.max() or 1), caption="|∇| de la región: dónde está la estructura fina", use_container_width=True, clamp=True)
    st.caption("Lo que ves en la región y NO en el subrogado (filamentos continuos, bordes nítidos, formas) es "
               "exactamente lo que los tests intentan cuantificar.")


# ---------------------------------------------------------------- alerta de descubrimiento

def _discovery_card(R):
    import json as _json

    from discovery import evaluate
    refs = []
    for up in st.session_state.get("ref_jsons") or []:
        try:
            refs.append(_json.loads(up.getvalue().decode("utf-8")))
        except Exception:
            continue
    g = evaluate(R["card"], masked=R.get("masked"), replicate=R.get("replicate"), references=refs or None)
    colors = {"invalid": "#5a1a1a", "none": "#1a1f1b", "unconfirmed": "#4a3210", "explained": "#3d3a10",
              "robust": "#0f3d1c", "pioneer": "#2e1747"}
    st.markdown(
        "<div style='background:%s;border:1px solid #33ff66;padding:0.8rem 1rem;margin:0.4rem 0 0.8rem 0'>"
        "<div style='font-size:1.4rem'>%s <b>%s</b></div><div>%s</div></div>"
        % (colors[g["level"]], g["icon"], g["title"], g["meaning"]), unsafe_allow_html=True)
    with st.expander("Comprobaciones de la alerta (%d de %d superadas)" % (sum(c["ok"] for c in g["checks"]), len(g["checks"])),
                     expanded=g["level"] in ("robust", "pioneer", "unconfirmed")):
        for c in g["checks"]:
            st.markdown("%s **%s** — %s" % ("✅" if c["ok"] else "❌", c["check"], c["detail"]))
        st.caption("Una comprobación que no se pudo hacer cuenta como no superada. Ningún nivel afirma un "
                   "descubrimiento: el más alto pide revisión experta y datos independientes (otro filtro, otra época, "
                   "otro instrumento).")
    return g


def _tab_golden(R):
    gw = R.get("golden")
    if not gw:
        st.info("Vuelve a pulsar Analizar para calcular la ventana φ.")
        return
    st.markdown("**Ventana Fibonacci / razón áurea** — φ = 1.6180…, ángulo áureo = 137.508°. "
                "Alerta solo si p < %.3f (Bonferroni para 2 tests al 1 %%)." % gw["alpha"])
    if gw["alert"]:
        st.success("🌻 Alerta φ: %s significativo frente a su nulo. Un p pequeño dice 'no es azar de este tipo', no "
                   "identifica el mecanismo." % ", ".join(gw["alert_tests"]))
    else:
        st.info("Sin evidencia de patrón φ frente a los nulos.")
    a, b = st.columns(2)
    for col, t in zip((a, b), gw["tests"]):
        col.markdown("**%s**" % t["test"])
        if t.get("p") is None:
            col.caption(t.get("note", ""))
            continue
        col.metric("p", "%.4f" % t["p"], "alerta" if t.get("alert") else "sin alerta", delta_color="off")
        col.caption(t.get("note", ""))
        if t.get("null"):
            obs = t.get("statistic", t.get("n_phi_pairs"))
            fig = go.Figure(go.Histogram(x=t["null"], nbinsx=30, marker_color=DIM, name="nulo"))
            fig.add_vline(x=obs, line_color=ACCENT, line_width=2)
            col.plotly_chart(_layout(fig, height=240, title="Observado (naranja) frente al nulo"), use_container_width=True)
        if t["test"].startswith("ángulo"):
            col.caption("%d fuentes puntuales usadas." % t["n_points"])
        else:
            col.caption("%d picos sobre la ley de potencia; %d pares con precisión suficiente; %d pares φ."
                        % (t["n_peaks"], t.get("n_resolvable_pairs", 0), t.get("n_phi_pairs", 0)))


def _tab_novelty():
    st.markdown("**Novedad frente a tus análisis anteriores**")
    st.caption("Sube los JSON de análisis previos (mínimo 5). Para cada descriptor se calcula cuánto se aleja este "
               "análisis de la mediana de tus referencias (z robusto). Con |z| ≥ 5 y una estructura robusta, la alerta "
               "sube a 🟣. Dice 'distinto de lo que ya analizaste', no 'nuevo para la ciencia'.")
    st.file_uploader("JSON de referencia", type=["json"], accept_multiple_files=True, key="ref_jsons")


# ---------------------------------------------------------------- entrada

def render_results(R, crop, name, x0, y0, side, n_null):
    mat, fdr, mc, prov, A, details = R["materials"], R["fdr"], R["mc"], R["prov"], R["analytics"], R["details"]
    gate = _discovery_card(R) if R.get("card") else None
    if mat.get("state") == "invalid_region":
        st.error("**Resultado NO VÁLIDO** — %s Las pestañas se muestran solo como diagnóstico." % mat.get("verdict"))
    else:
        st.markdown("**%s** — %s" % (mat.get("state"), mat.get("verdict")))
    k1, k2, k3 = st.columns(3)
    k4, k5, k6 = st.columns(3)
    k1.metric("Familia dominante", mat.get("dominant_label") or "—")
    k2.metric("FDR", "%d / %d pasan" % (fdr.get("n_passed", 0), fdr.get("n_tested", 0)))
    k3.metric("p score global", "%.3f" % mc.get("p_value", 1.0), "z %+.2f" % mc.get("z_score", 0), delta_color="off")
    top = max(details, key=lambda k: details[k]["z"]) if details else None
    k4.metric("Mayor desviación", top or "—", "z %+.2f" % details[top]["z"] if top else None, delta_color="off")
    beta = A["spectrum"]["fit"]["beta"]
    k5.metric("β espectral", "%.2f" % beta if np.isfinite(beta) else "—", "R² %.2f" % A["spectrum"]["fit"]["r2"], delta_color="off")
    k6.metric("Procedencia", "%s" % prov.get("verdict"), "%.2f" % float(prov.get("trust_score") or 0), delta_color="off")
    if fdr.get("can_pass") is False:
        st.warning("Con %d subrogados el FDR no puede pasar: sube el número en ajustes avanzados." % n_null)
    if mat.get("point_source_warning"):
        st.warning("**%d fuentes puntuales** en la región (estrellas o galaxias no resueltas). Medido con imágenes "
                   "simuladas: solo ruido → 0 descriptores pasan el FDR; el mismo ruido con 14 fuentes puntuales → de 1 a 4. "
                   "Un FDR que pasa aquí puede deberse a las fuentes, no a estructura extendida (nube, filamento). "
                   "Mira el Mapa local: si la señal se concentra en pocas teselas, son las fuentes." % mat["n_point_sources"])
    if mat.get("instrument_warning_reason"):
        st.warning("Aviso de instrumento: %s" % mat["instrument_warning_reason"])

    tabs = st.tabs(["Nulo y FDR", "Espectro", "Escalas", "Mapa local", "Familias", "Descriptores", "Región vs nulo",
                    "🌻 Fibonacci / φ", "Novedad"])
    with tabs[0]:
        _tab_null(R, n_null)
    with tabs[1]:
        _tab_spectrum(A)
    with tabs[2]:
        _tab_scales(A)
    with tabs[3]:
        _tab_local(A, crop)
    with tabs[4]:
        _tab_families(R)
    with tabs[5]:
        df = _tab_descriptors(R)
    with tabs[6]:
        _tab_compare(crop)
    with tabs[7]:
        _tab_golden(R)
    with tabs[8]:
        _tab_novelty()
        if gate and gate.get("novelty") and gate["novelty"].get("available"):
            nv = gate["novelty"]
            st.metric("Máximo |z| frente a %d referencias" % nv["n_references"], "%.1f" % (nv["max_abs_z"] or 0))
            st.dataframe([{"descriptor": k, "valor": v["value"], "mediana referencias": v["ref_median"], "z": v["z"]}
                          for k, v in nv["per_descriptor"].items()], hide_index=True, use_container_width=True)
        elif gate and gate.get("novelty"):
            st.caption(gate["novelty"].get("note", ""))

    stem = "cms80_%s_x%d_y%d_s%d" % (str(name).split(".")[0], x0, y0, side)
    d1, d2, d3 = st.columns(3)
    d1.download_button("Descargar informe JSON", R["json"], file_name=stem + ".json")
    tests = pd.DataFrame([{"descriptor": k, **{kk: vv for kk, vv in d.items() if kk != "null"}} for k, d in details.items()])
    d2.download_button("Tests vs nulo (CSV)", tests.to_csv(index=False), file_name=stem + "_tests.csv")
    if df is not None:
        d3.download_button("Descriptores (CSV)", df.to_csv(index=False), file_name=stem + "_descriptores.csv")
