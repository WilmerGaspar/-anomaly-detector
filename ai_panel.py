"""🤖 IA conectadas: claves, semáforo por IA y consejo, dentro del recuadro 🧠 Guía.

Dos partes:
- `render_controls` (columna derecha de la guía): semáforo, pegar una clave, preguntas de un clic.
- `render_council` (a lo ancho, debajo): las respuestas aparecen según llegan, cada una con su
  comprobación, el consenso con la acción propuesta y el historial de preguntas.

Claves: se pueden pegar en la app (solo viven en esta sesión del navegador: no se guardan ni se
muestran) o dejarlas fijas en Streamlit → Settings → Secrets. Las IA no ejecutan nada: proponen
una acción y la persona la acepta.
"""
import os
import time

import streamlit as st

import ai_hub as H

ACTION_LABELS = {"auto": "▶ Hazlo por mí (buscar, cargar y analizar)", "auto_analyze": "▶ Analizar por mí",
                 "other_filter": "▶ Otro archivo del mismo objeto"}
MAX_HISTORY = 6

SECRETS_STEPS = (
    "**Para dejarla fija (no hay que pegarla cada vez):**\n"
    "1. Entra en **share.streamlit.io** con la cuenta con la que publicaste la app.\n"
    "2. En la lista de apps, a la derecha de la tuya, pulsa **⋮** → **Settings**. (Con la app abierta y tu "
    "sesión iniciada también sirve el botón **Manage app**, abajo a la derecha → **⋮** → **Settings**.)\n"
    "3. Pestaña **Secrets**: pega una línea por clave y pulsa **Save changes**:\n"
    "```\nNVIDIA_API_KEY = \"nvapi-…\"\nOPENROUTER_API_KEY = \"sk-or-…\"\n```\n"
    "4. Recarga la app: el semáforo pasa a 🟢.\n\n"
    "Otras: `GROQ_API_KEY` (gsk_…), `GEMINI_API_KEY` (AIza…). Con la de OpenRouter se encienden dos IA "
    "(OpenRouter y DeepSeek). Modelos opcionales: `NVIDIA_MODEL`, `GROQ_MODEL`, `OPENROUTER_MODEL`, "
    "`DEEPSEEK_MODEL`, `GEMINI_MODEL`.\n\n"
    "**OpenRouter, siempre gratis:** la app usa `openrouter/free` (su enrutador oficial elige un modelo gratis en "
    "cada pregunta) y un DeepSeek que la lista pública de OpenRouter marque gratis ese día (precio 0 y `:free`). "
    "OpenRouter retira modelos gratis a menudo; si uno responde «unavailable for free», la app pasa al siguiente "
    "gratis y nunca al de pago. Los gratis solo funcionan si en **openrouter.ai/settings/privacy** permites los "
    "modelos gratuitos (pueden guardar las preguntas; la app solo envía números públicos del telescopio). Si no, "
    "salen 🔴 con «data policy»."
)

CSS = """<style>
.ai-chip{display:inline-block;padding:0.1rem 0.55rem;margin:0.12rem 0.2rem 0.12rem 0;border:1px solid #2a5a35;
 border-radius:999px;font-size:0.82rem;white-space:nowrap}
.ai-chip.on{border-color:#33ff66;color:#33ff66}.ai-chip.err{border-color:#ff6b6b;color:#ff9b9b}
.ai-chip.wait{border-color:#ffb347;color:#ffcf8a}
.ai-head{font-size:0.85rem;color:#8fd9a0}.ai-ok{color:#33ff66}.ai-warn{color:#ffb347}
</style>"""


# ---------------------------------------------------------------- claves

def _session_keys():
    return st.session_state.setdefault("ai_session_keys", {})


def _session_models():
    return st.session_state.setdefault("ai_session_models", {})


def get_secret(name):
    """Clave (o modelo) puesto en esta sesión; si no, Secrets de Streamlit; si no, variable de entorno."""
    if name in _session_keys():
        return _session_keys()[name]
    if _session_models().get(name):
        return _session_models()[name]
    try:
        if st.secrets.load_if_toml_exists():
            value = st.secrets.get(name)
            if value:
                return str(value)
    except Exception:
        pass
    return os.environ.get(name)


def _key_source(provider):
    return "pegada en esta sesión" if provider["key"] in _session_keys() else "de Secrets"


def _key_form():
    """Pegar una clave: se reconoce el proveedor por cómo empieza. Solo vive en esta sesión."""
    names = [p["key"] for p in H.PROVIDERS if p["id"] != "deepseek"]
    labels = {n: " + ".join(H.providers_for_key(n)) for n in names}
    with st.form("ai_key_form", clear_on_submit=True):
        key = st.text_input("Pega aquí tu clave API", type="password",
                            placeholder="nvapi-…  ·  sk-or-…  ·  gsk_…  ·  AIza…")
        choice = st.selectbox("¿De qué servicio es? (se reconoce sola por cómo empieza)",
                              ["Detectar automáticamente"] + names,
                              format_func=lambda n: n if n == "Detectar automáticamente" else labels[n])
        sent = st.form_submit_button("🔌 Conectar", use_container_width=True)
    if sent:
        key = (key or "").strip()
        name = H.detect_key_name(key) if choice == "Detectar automáticamente" else choice
        if not key:
            st.warning("Pega la clave en el campo y pulsa 🔌 Conectar.")
        elif not name:
            st.warning("No reconozco el servicio por cómo empieza la clave: elígelo en la lista y vuelve a pulsar.")
        else:
            _session_keys()[name] = key
            st.session_state.pop("ai_status", None)        # se vuelve a probar con la clave nueva
            st.session_state["ai_flash"] = "Clave de %s guardada para esta sesión: probada abajo en el semáforo." % labels[name]
            st.rerun()
    st.caption("🔒 La clave pegada aquí solo vive en esta sesión del navegador: no se guarda en ningún archivo, no se "
               "muestra y se borra al cerrar la pestaña. Nunca la pegues en un chat ni en GitHub.")
    if _session_keys() and st.button("Olvidar las claves de esta sesión", key="ai_forget"):
        st.session_state["ai_session_keys"] = {}
        st.session_state.pop("ai_status", None)
        st.rerun()


# ---------------------------------------------------------------- semáforo

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


def _chips(providers, status):
    cls = {H.CONNECTED: "on", H.ERROR: "err", H.UNTESTED: "wait", H.NO_KEY: ""}
    html = "".join('<span class="ai-chip %s">%s %s</span>' % (cls[status.get(p["id"], {}).get("state", H.NO_KEY)],
                                                             H.LIGHT[status.get(p["id"], {}).get("state", H.NO_KEY)],
                                                             p["name"]) for p in providers)
    st.markdown(html, unsafe_allow_html=True)


def _error_lines(providers, status):
    """El motivo de cada 🔴 a la vista (antes quedaba dentro de «Detalle»). Las IA que comparten clave y
    error (OpenRouter y DeepSeek) van en una sola línea."""
    groups = {}
    for p in providers:
        s = status.get(p["id"], {})
        if s.get("state") == H.ERROR:
            groups.setdefault(s.get("detail") or "error", []).append(p["name"])
    for detail, names in groups.items():
        st.warning("🔴 **%s**: %s" % (" y ".join(names), detail))


# ---------------------------------------------------------------- columna derecha

def render_controls(question, card, step, log, gate):
    """Semáforo, claves y preguntas. Una pregunta queda pendiente y la responde `render_council`."""
    ss = st.session_state
    st.markdown(CSS, unsafe_allow_html=True)
    providers = H.configured(get_secret)
    keyed = [p for p in providers if p["api_key"]]
    st.markdown("**🤖 IA conectadas**")
    if ss.get("ai_flash"):
        st.success(ss.pop("ai_flash"))
    status = _status(providers, force=st.session_state.pop("ai_force_ping", False))
    _chips(providers, status)
    _error_lines(providers, status)
    with st.expander("🔑 Conectar una IA (pegar la clave)", expanded=not keyed):
        _key_form()
        st.markdown(SECRETS_STEPS)
    if keyed:
        with st.expander("Detalle de cada IA", expanded=any(s.get("state") == H.ERROR for s in status.values())):
            for p in providers:
                line = H.status_detail(p, status.get(p["id"], {}))
                if p["api_key"]:
                    line += " · clave %s" % _key_source(p)
                st.markdown(line + " · [web](%s)" % p["signup"])
            # Cambiar de modelo sin ir a Secrets (si uno gratuito desaparece): solo esta sesión.
            with st.form("ai_model_form"):
                st.caption("Cambiar el modelo (solo esta sesión; vacío = el de la app, gratis). En OpenRouter "
                           "usa solo nombres que terminen en `:free` u `openrouter/free`: los demás son de pago.")
                new_models = {p["model_key"]: st.text_input(p["name"], value=_session_models().get(p["model_key"], ""),
                                                            placeholder=p["model"], key="ai_model_" + p["id"])
                              for p in providers if p["api_key"]}
                if st.form_submit_button("Usar estos modelos"):
                    _session_models().update({k: v.strip() for k, v in new_models.items()})
                    ss.pop("ai_status", None)
                    st.rerun()
            if st.button("🔄 Probar conexiones", key="ai_ping"):
                ss["ai_force_ping"] = True
                st.rerun()
    else:
        st.caption("Sin IA conectada la guía por reglas funciona igual, gratis y sin conexión.")
        return
    actions = tuple(step.get("actions") or ())
    context = H.build_context(card, log, step, actions, gate=gate)
    st.caption("Preguntas rápidas:")
    for i, (label, q) in enumerate(H.QUICK_QUESTIONS):
        if st.button(label, key="ai_quick_%d" % i, use_container_width=True):
            ss["ai_pending"] = {"q": q, "ctx": context}
    if st.button("🤖 Preguntar a las IA (%d)" % len(keyed), key="ai_ask", type="primary", use_container_width=True,
                 help="Envía tu pregunta de arriba (o, si está vacía, «%s») con el resumen del análisis."
                      % H.DEFAULT_QUESTION):
        ss["ai_pending"] = {"q": (question or "").strip() or H.DEFAULT_QUESTION, "ctx": context}
    ss["ai_context_now"] = context


# ---------------------------------------------------------------- consejo (a lo ancho)

def _readable(text):
    """Cada apartado («**Qué muestra:**»…) en su párrafo y los saltos de línea respetados."""
    import re
    text = re.sub(r"\s*(\*\*(?:Qué|Siguiente)[^*]{0,40}:\*\*)", r"\n\n\1", str(text or "")).strip()
    return re.sub(r"(?<!\n)\n(?!\n)", "  \n", text)


def _answer_card(r):
    with st.container(border=True):
        model = r.get("model") or ""
        if r.get("served_model") and r["served_model"] != model:
            model += " → " + r["served_model"]                # qué modelo gratis eligió openrouter/free
        meta = " · ".join(x for x in (model, ("%.1f s" % (r["latency_ms"] / 1000.0))
                                      if r.get("latency_ms") else "") if x)
        if not r["ok"]:
            st.markdown("🔴 **%s** <span class='ai-head'>%s</span>" % (r["name"], meta), unsafe_allow_html=True)
            st.caption(r["detail"])
            return
        badge = ("<span class='ai-ok'>✅ pasa la comprobación</span>" if not r["issues"]
                 else "<span class='ai-warn'>⚠️ revísala</span>")
        st.markdown("🤖 **%s** · %s<br><span class='ai-head'>%s</span>" % (r["name"], badge, meta),
                    unsafe_allow_html=True)
        st.markdown(_readable(r["text"]) or "_(sin texto)_")
        if r["issues"]:
            st.caption("⚠️ " + "; ".join(r["issues"]))
        if r["action"]:
            st.caption("Propone: " + ACTION_LABELS[r["action"]])


def _consensus(results, actions, current, on_action, key):
    ok = [r for r in results if r["ok"]]
    verified = [r for r in ok if not r["issues"]]
    action, votes, n_ok = H.consensus(results)
    st.markdown("🧭 **Consenso:** respondieron %d de %d IA · %d %s la comprobación%s"
                % (len(ok), len(results), len(verified), "pasa" if len(verified) == 1 else "pasan",
                   (" · proponen %s (%d)" % (ACTION_LABELS[action], votes)) if action else ""))
    if not current:
        st.caption("Estas respuestas son sobre un paso anterior: vuelve a preguntar para actualizarlas.")
    elif action and action in actions:
        if st.button("✅ Aceptar la propuesta (%d de %d IA): %s" % (votes, n_ok, ACTION_LABELS[action]),
                     key=key, type="primary", use_container_width=True):
            on_action(action)
    elif n_ok:
        st.caption("Las IA no proponen ninguna acción disponible ahora.")


def _grid(results, n_cols=2):
    cols = st.columns(min(n_cols, max(1, len(results))))
    for i, r in enumerate(results):
        with cols[i % len(cols)]:
            _answer_card(r)


def render_council(step, on_action):
    """Respuestas de las IA a lo ancho: en vivo mientras llegan y después el historial."""
    ss = st.session_state
    pending = ss.pop("ai_pending", None)
    providers = H.configured(get_secret)
    actions = tuple(step.get("actions") or ())
    context = ss.get("ai_context_now")
    history = ss.setdefault("ai_history", [])
    if pending:
        active = [p for p in providers if p["api_key"]]
        st.markdown("#### 🤖 Consejo de IA · «%s»" % pending["q"])
        cols = st.columns(min(2, max(1, len(active))))
        slots = {p["id"]: cols[i % len(cols)].empty() for i, p in enumerate(active)}
        for p in active:
            slots[p["id"]].info("⏳ %s está pensando…%s" % (p["name"], " (razona antes: puede tardar)"
                                                          if H.is_reasoning(p) else ""))
        results = []
        for r in H.ask_iter(providers, pending["q"], pending["ctx"]):
            results.append(r)
            with slots[r["id"]].container():
                _answer_card(r)
        order = {p["id"]: i for i, p in enumerate(providers)}
        results.sort(key=lambda r: order[r["id"]])
        history.insert(0, {"q": pending["q"], "ctx": pending["ctx"], "results": results,
                           "time": time.strftime("%H:%M")})
        del history[MAX_HISTORY:]
        cur = ss.get("ai_status") or {"fp": H.fingerprint(providers), "by_id": H.initial_status(providers)}
        ss["ai_status"] = {"fp": cur["fp"], "by_id": H.status_after_answers(cur["by_id"], results)}
        st.rerun()                                     # semáforo y consenso con todo ya llegado
    if not history:
        return
    latest = history[0]
    st.markdown("#### 🤖 Consejo de IA · «%s» · %s" % (latest["q"], latest["time"]))
    _consensus(latest["results"], actions, latest["ctx"] == context, on_action, key="ai_do")
    _grid(latest["results"])
    st.caption("IA externas: pueden equivocarse. El semáforo, los números y la tabla de hipótesis los calcula la app. "
               "Cada respuesta se comprueba: cifras que no están en los datos, referencias inventadas, datos citados "
               "que no existen y afirmaciones de más (✅/⚠️). Solo las ✅ votan la acción.")
    for i, h in enumerate(history[1:], 1):
        n_ok = sum(1 for r in h["results"] if r["ok"] and not r["issues"])
        with st.expander("Antes · %s · «%s» · %d ✅" % (h["time"], h["q"], n_ok)):
            if h["ctx"] != context:
                st.caption("Sobre un paso anterior del análisis.")
            _grid(h["results"])
