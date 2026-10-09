"""🤖 IA conectadas: semáforo por IA y consejo, dentro de la 🧠 Guía de la barra lateral.

Las claves se leen de Streamlit → Settings → Secrets (o variables de entorno) y nunca se
muestran. Sin claves solo aparece el semáforo en ⚪ y cómo conectarlas: la guía por reglas
sigue funcionando igual. Las IA no ejecutan nada: proponen una acción y la persona la acepta.
"""
import os

import streamlit as st

import ai_hub as H

ACTION_LABELS = {"auto": "▶ Hazlo por mí (buscar, cargar y analizar)", "auto_analyze": "▶ Analizar por mí",
                 "other_filter": "▶ Otro archivo del mismo objeto"}

HOW_TO = (
    "**Cómo conectar una IA (gratis):**\n"
    "1. Crea una cuenta y una clave API en la web del proveedor (enlaces arriba).\n"
    "2. En Streamlit Cloud: tu app → ⋮ → **Settings** → **Secrets** y pega una línea por clave, por ejemplo  \n"
    "`NVIDIA_API_KEY = \"nvapi-…\"`\n"
    "3. Guarda y recarga la página: si la clave es correcta, el semáforo pasa a 🟢 (puede tardar un minuto).\n\n"
    "Nombres: `NVIDIA_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY`. **DeepSeek R1** usa la "
    "misma clave de OpenRouter (gratis): con `OPENROUTER_API_KEY` se encienden las dos. Opcional, para cambiar de "
    "modelo sin tocar código: `NVIDIA_MODEL`, `GROQ_MODEL`, `OPENROUTER_MODEL`, `DEEPSEEK_MODEL`, `GEMINI_MODEL`. "
    "DeepSeek R1 piensa antes de responder: tarda más (hasta 2-3 minutos).\n\n"
    "Nunca pegues una clave en un chat ni en GitHub. A las IA solo se les envía el resumen del análisis "
    "(números públicos del telescopio) y tu pregunta."
)


def get_secret(name):
    """Secrets de Streamlit y, si no hay, variable de entorno. Sin archivo de secrets no falla."""
    try:
        if st.secrets.load_if_toml_exists():
            value = st.secrets.get(name)
            if value:
                return str(value)
    except Exception:
        pass
    return os.environ.get(name)


def _status(providers, force=False):
    """Semáforo de la sesión. Se prueba una vez al abrir, al pulsar 🔄 o si cambian las claves."""
    ss = st.session_state
    fp = H.fingerprint(providers)
    cur = ss.get("ai_status")
    if force or not cur or cur.get("fp") != fp:
        status = H.initial_status(providers)
        keyed = [p for p in providers if p["api_key"]]
        if keyed:
            with st.spinner("Probando la conexión con %d IA…" % len(keyed)):
                status.update(H.ping_all(keyed))
        cur = {"fp": fp, "by_id": status}
        ss["ai_status"] = cur
    return cur["by_id"]


def render(question, card, step, log, gate, on_action):
    """Dibuja el panel (dentro de `with st.sidebar`). `on_action(a)` ejecuta una acción de la guía."""
    ss = st.session_state
    providers = H.configured(get_secret)
    keyed = [p for p in providers if p["api_key"]]
    st.markdown("**🤖 IA conectadas**")
    force = st.button("🔄 Probar conexiones", key="ai_ping", disabled=not keyed)
    status = _status(providers, force=force)
    st.markdown(H.light_line(providers, status))
    with st.expander("Detalle de cada IA y cómo conectarlas"):
        for p in providers:
            st.markdown(H.status_detail(p, status.get(p["id"], {})) + " · [web](%s)" % p["signup"])
        st.caption("⚪ sin clave · 🟡 clave puesta, sin probar · 🟢 conectada · 🔴 error (el motivo sale al lado)")
        st.markdown(HOW_TO)
    if not keyed:
        st.caption("Ninguna IA conectada: la guía por reglas funciona igual, gratis y sin conexión.")
        return
    actions = tuple(step.get("actions") or ())
    context = H.build_context(card, log, step, actions, gate=gate)
    ask = st.button("🤖 Preguntar a las IA (%d)" % len(keyed), key="ai_ask", use_container_width=True,
                    help="Envía tu pregunta (o, si está vacía, «%s») y el resumen del análisis." % H.DEFAULT_QUESTION)
    if ask:
        q = (question or "").strip() or H.DEFAULT_QUESTION
        with st.spinner("Consultando a %d IA a la vez…" % len(keyed)):
            results = H.ask_all(providers, q, context)
        ss["ai_answers"] = {"q": q, "ctx": context, "results": results}
        ss["ai_status"] = {"fp": H.fingerprint(providers), "by_id": H.status_after_answers(status, results)}
        st.rerun()                                     # el semáforo de arriba refleja la consulta
    ans = ss.get("ai_answers")
    if ans:
        _render_answers(ans, context, actions, on_action)


def _render_answers(ans, context, actions, on_action):
    st.markdown("**Consejo de IA** · «%s»" % ans["q"])
    for r in ans["results"]:
        if not r["ok"]:
            st.caption("🔴 %s: %s" % (r["name"], r["detail"]))
            continue
        badge = "✅ pasa la comprobación" if not r["issues"] else "⚠️ " + "; ".join(r["issues"])
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown("**%s** · `%s` · %s" % (r["name"], r["model"], badge))
            st.markdown(r["text"] or "_(sin texto)_")
            if r["action"]:
                st.caption("Propone: " + ACTION_LABELS[r["action"]])
    st.caption("IA externa: puede equivocarse. El semáforo y los números los calcula la app; se comprueba que "
               "cada respuesta no invente cifras ni afirme descubrimientos (✅/⚠️). Verifícalo tú también.")
    if ans["ctx"] != context:
        st.caption("Estas respuestas son sobre un paso anterior: vuelve a preguntar para actualizarlas.")
        return
    action, votes, n_ok = H.consensus(ans["results"])
    if action and action in actions:
        if st.button("✅ Aceptar la propuesta (%d de %d IA): %s" % (votes, n_ok, ACTION_LABELS[action]),
                     key="ai_do", type="primary", use_container_width=True):
            on_action(action)
    elif n_ok:
        st.caption("Las IA no proponen ninguna acción disponible ahora.")
