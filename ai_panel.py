"""🤖 IA conectadas: claves, semáforo por IA y consejo, dentro del recuadro 🧠 Guía.

Dos partes:
- `render_controls` (columna derecha de la guía): semáforo, pegar una clave, preguntas de un clic.
- `render_council` (a lo ancho, debajo): las respuestas aparecen según llegan, cada una con su
  comprobación, el consenso con la acción propuesta y el historial de preguntas.

Claves: se pueden pegar en la app (solo viven en esta sesión del navegador: no se guardan ni se
muestran) o dejarlas fijas en Streamlit → Settings → Secrets. Las IA no ejecutan nada: proponen
una acción y la persona la acepta.
"""
import html
import os
import re
import time

import streamlit as st

import ai_hub as H
import theme

ACTION_LABELS = {"auto": "▶ Hazlo por mí (buscar, cargar y analizar)", "auto_analyze": "▶ Analizar por mí",
                 "other_filter": "▶ Otro archivo del mismo objeto",
                 "more_null": "▶ Repetir con más subrogados (p más preciso)",
                 "known_control": "▶ Analizar un control conocido (HST · Crab Nebula)"}
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
    "**OpenRouter, siempre gratis:** la app usa el modelo gratis más capaz del día de la lista pública de OpenRouter "
    "(precio 0 y `:free`; los muy pequeños, al final; su enrutador `openrouter/free`, de último recurso) y un "
    "DeepSeek que esa lista marque gratis. "
    "OpenRouter retira modelos gratis a menudo; si uno responde «unavailable for free», la app pasa al siguiente "
    "gratis y nunca al de pago. Los gratis solo funcionan si en **openrouter.ai/settings/privacy** permites los "
    "modelos gratuitos (pueden guardar las preguntas; la app solo envía números públicos del telescopio). Si no, "
    "salen 🔴 con «data policy»."
)

# Estilos (chips, consola, tarjetas): en theme.py, con el resto del aspecto.


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
        keyed = [p for p in providers if H.usable(p)]          # las 💤 no se prueban: hoy no hay modelo gratis
        if keyed:
            with st.spinner("Probando la conexión con %d IA…" % len(keyed)):
                status.update(H.ping_all(keyed))
        cur = {"fp": fp, "by_id": status}
        ss["ai_status"] = cur
    return cur["by_id"]


def _chips(providers, status):
    cls = {H.CONNECTED: "on", H.ERROR: "err", H.UNTESTED: "wait", H.NO_KEY: "", H.RESTING: "off"}
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
    for p in providers:                                # 💤: aviso tranquilo, no es un error que arreglar
        s = status.get(p["id"], {})
        if s.get("state") == H.RESTING:
            st.caption("💤 **%s**: %s." % (p["name"], s.get("detail") or H.STATE_TEXT[H.RESTING]))


# ---------------------------------------------------------------- columna derecha

def render_controls(question, card, step, log, gate, flow_items=None, feasible=()):
    """Semáforo, claves y preguntas. Una pregunta queda pendiente y la responde `render_council`.
    `flow_items`/`feasible`: revisión del flujo por reglas y órdenes posibles (flow.py)."""
    ss = st.session_state
    providers = H.configured(get_secret)
    keyed = [p for p in providers if p["api_key"]]
    st.markdown("**🤖 IA conectadas**")
    if ss.get("ai_flash"):
        st.success(ss.pop("ai_flash"))
    status = _status(providers, force=st.session_state.pop("ai_force_ping", False))
    ss["ai_tray"] = [(H.LIGHT[status.get(p["id"], {}).get("state", H.NO_KEY)], p["name"]) for p in keyed]
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
        ss.pop("flow_review_request", None)
        return
    ready = [p for p in keyed if H.usable(p)]
    if not ready:
        st.caption("Ninguna IA tiene hoy un modelo gratis: la guía por reglas funciona igual.")
        ss.pop("flow_review_request", None)
        return
    actions = tuple(dict.fromkeys(tuple(step.get("actions") or ()) + tuple(feasible)))
    context = H.build_context(card, log, step, actions, gate=gate, flow=flow_items)
    # Ventana «¿Listo para trabajar?»: una vez por sesión, cuando al menos una IA está 🟢.
    if ss.get("ai_welcome") is None and any(status.get(p["id"], {}).get("state") == H.CONNECTED for p in ready):
        ss["ai_welcome"] = "open"
    if ss.get("ai_welcome") == "open":
        _welcome([p["name"] for p in ready], flow_items or [])
    if ss.pop("flow_review_request", False):
        ss["ai_pending"] = {"q": H.REVIEW_QUESTION, "ctx": context, "kind": "orders"}
    st.caption("Preguntas rápidas:")
    for i, (label, q) in enumerate(H.QUICK_QUESTIONS):
        if st.button(label, key="ai_quick_%d" % i, use_container_width=True):
            ss["ai_pending"] = {"q": q, "ctx": context}
    if st.button("🤖 Preguntar a las IA (%d)" % len(ready), key="ai_ask", type="primary", use_container_width=True,
                 help="Envía tu pregunta de arriba (o, si está vacía, «%s») con el resumen del análisis."
                      % H.DEFAULT_QUESTION):
        ss["ai_pending"] = {"q": (question or "").strip() or H.DEFAULT_QUESTION, "ctx": context}
    ss["ai_context_now"] = context


# ---------------------------------------------------------------- consejo (a lo ancho)
# Ventana «Consejo de IA»: marco Windows 2000 y, dentro, una consola oscura con un módulo y una luz por IA,
# barra de progreso de cuadritos mientras piensan, cada dato citado marcado ✓/✗ y la barra del consenso.

LED = {H.CONNECTED: "on", H.ERROR: "err", H.UNTESTED: "wait", H.NO_KEY: "", H.RESTING: "zz"}


def _short(model):
    """«google/gemma-4-31b-it:free» → «gemma-4-31b-it» (el nombre completo sigue en «Detalle»)."""
    return str(model or "").split("/")[-1].split(":")[0]


def _readable(text):
    """Cada apartado («**Qué muestra:**»…) en su párrafo y los saltos de línea respetados."""
    text = re.sub(r"\s*(\*\*(?:Qué|Siguiente)[^*]{0,40}:\*\*)", r"\n\n\1", str(text or "")).strip()
    return re.sub(r"(?<!\n)\n(?!\n)", "  \n", text)


def _answer_html(text, known):
    """Texto de la IA sin HTML propio (se escapa: es texto de fuera) y con cada dato citado marcado:
    ✓ si existe en tus datos, ✗ tachado si no. Las imágenes no se cargan."""
    esc = html.escape(str(text or ""), quote=False).replace("![", "!​[")

    def mark(m):
        ok = H.cite_ok(m.group(1), known)
        return '<span class="cite %s">%s %s</span>' % ("ok" if ok else "bad", m.group(1), "✓" if ok else "✗")
    return _readable(H._CITE.sub(mark, esc))


CHECKS = (("números", "cita números"), ("referencias", "cita referencias"), ("datos citados", "cita datos"),
          ("sin afirmar de más", "afirma de más"))


def _checks_html(issues):
    out = []
    for label, prefix in CHECKS:
        bad = any(i.startswith(prefix) for i in issues)
        out.append('<span class="%s">%s %s</span>' % ("n" if bad else "y", "✗" if bad else "✓", label))
    return '<div class="hud-checks">%s</div>' % "".join(out)


def _answer_card(r, known=None, key=""):
    with st.container(key="aicard_%s_%s" % (key, r["id"])):
        model = _short(r.get("model"))
        if r.get("served_model") and _short(r["served_model"]) != model:
            model += " → " + _short(r["served_model"])          # qué modelo gratis eligió el enrutador
        meta = " · ".join(x for x in (model, ("%.1f s" % (r["latency_ms"] / 1000.0))
                                      if r.get("latency_ms") else "") if x)
        if not r["ok"]:
            st.markdown("<div class='hud-head'>🔴 ▌ %s <span class='m'>%s</span></div>"
                        % (html.escape(r["name"].upper()), html.escape(meta)), unsafe_allow_html=True)
            st.caption(r["detail"])
            return
        badge = ("<span class='ai-ok'>✅ pasa la comprobación</span>" if not r["issues"]
                 else "<span class='ai-warn'>⚠️ revísala</span>")
        st.markdown("<div class='hud-head'>🤖 ▌ %s · %s<br><span class='m'>%s</span></div>"
                    % (html.escape(r["name"].upper()), badge, html.escape(meta)), unsafe_allow_html=True)
        st.markdown(_answer_html(r["text"], known) or "_(sin texto)_", unsafe_allow_html=True)
        st.markdown(_checks_html(r["issues"]), unsafe_allow_html=True)
        if r["issues"]:
            st.caption("⚠️ " + "; ".join(r["issues"]))
        if r["action"]:
            st.caption("Propone: " + ACTION_LABELS[r["action"]])


def _modules(providers, status, results=()):
    """Un módulo por IA con su luz: modelo, tiempo y si su última respuesta pasó la comprobación."""
    by = {r["id"]: r for r in results}
    cells = []
    for p in providers:
        s = status.get(p["id"], {})
        state = s.get("state", H.NO_KEY)
        r = by.get(p["id"])
        if state == H.NO_KEY:
            sub = "sin clave"
        elif state == H.RESTING:
            sub = "💤 sin modelo gratis hoy"
        elif r and r["ok"]:
            sub = "%s · %.1f s · %s" % (_short(r.get("served_model") or r.get("model")), (r["latency_ms"] or 0) / 1000.0,
                                       "✓ verificada" if not r["issues"] else "⚠ revisar")
        elif r:
            sub = "sin respuesta"
        else:
            sub = "%s · %s" % (_short(s.get("model") or p["model"]), H.STATE_TEXT[state])
        cells.append('<div class="hud-mod"><span class="led %s"></span><b>%s</b><small>%s</small></div>'
                     % (LED[state], html.escape(p["name"]), html.escape(sub)))
    st.markdown('<div class="hud-mods">%s</div>' % "".join(cells), unsafe_allow_html=True)


def _consensus(results, actions, current, on_action, key):
    ok = [r for r in results if r["ok"]]
    verified = [r for r in ok if not r["issues"]]
    action, votes, n_ok = H.consensus(results)
    st.markdown("🧭 **Consenso:** respondieron %d de %d IA · %d %s la comprobación%s"
                % (len(ok), len(results), len(verified), "pasa" if len(verified) == 1 else "pasan",
                   (" · proponen %s (%d)" % (ACTION_LABELS[action], votes)) if action else ""))
    st.markdown('<div class="hud-cons"><div class="bar"><i style="width:%d%%"></i></div><span>%d de %d ✓</span></div>'
                % (round(100.0 * len(verified) / max(1, len(results))), len(verified), len(results)),
                unsafe_allow_html=True)
    if not current:
        st.caption("Estas respuestas son sobre un paso anterior: vuelve a preguntar para actualizarlas.")
    elif action and action in actions:
        if st.button("✅ Aceptar la propuesta (%d de %d IA): %s" % (votes, n_ok, ACTION_LABELS[action]),
                     key=key, type="primary", use_container_width=True):
            on_action(action)
    elif n_ok:
        st.caption("Las IA no proponen ninguna acción disponible ahora.")


def _grid(results, known=None, key="", n_cols=2):
    cols = st.columns(min(n_cols, max(1, len(results))))
    for i, r in enumerate(results):
        with cols[i % len(cols)]:
            _answer_card(r, known, key)


def _thinking(p):
    return ('<div class="hud-mod"><span class="led wait"></span><b>%s</b> pensando…%s'
            '<div class="hud-prog run"><span></span></div></div>'
            % (html.escape(p["name"]), " <small>(razona antes: puede tardar)</small>" if H.is_reasoning(p) else ""))


def render_council(step, on_action, feasible=()):
    """Respuestas de las IA a lo ancho: en vivo mientras llegan y después el historial."""
    ss = st.session_state
    pending = ss.pop("ai_pending", None)
    providers = H.configured(get_secret)
    actions = tuple(dict.fromkeys(tuple(step.get("actions") or ()) + tuple(feasible)))
    context = ss.get("ai_context_now")
    history = ss.setdefault("ai_history", [])
    status = (ss.get("ai_status") or {}).get("by_id") or H.initial_status(providers)
    if not pending and not history:
        return
    with st.container(key="win_ai"):
        theme.window_title("Consejo de IA — consola de interpretación", "🤖",
                           "trabajando…" if pending else "listo · %s" % history[0]["time"])
        with st.container(key="ai_console"):
            if pending:
                _council_live(pending, providers, status, ss, history)       # termina con st.rerun()
            latest = history[0]
            _modules(providers, status, latest["results"])
            n_ok = sum(1 for r in latest["results"] if r["ok"])
            st.markdown('<div class="hud-line">&gt; «%s» · %s · %d de %d IA respondieron</div>'
                        '<div class="hud-prog"><span style="--fill:%d%%"></span></div>'
                        % (html.escape(latest["q"]), latest["time"], n_ok, len(latest["results"]),
                           round(100.0 * n_ok / max(1, len(latest["results"])))), unsafe_allow_html=True)
            known = H.known_names(latest["ctx"])
            _grid(latest["results"], known, key="h0")
            _consensus(latest["results"], actions, latest["ctx"] == context, on_action, key="ai_do")
            st.caption("IA externas: pueden equivocarse. El semáforo, los números y la tabla de hipótesis los calcula la "
                       "app. Cada respuesta se comprueba: cifras que no están en los datos, referencias inventadas, datos "
                       "citados que no existen (✗ tachados) y afirmaciones de más (✅/⚠️). Solo las ✅ votan la acción.")
            for i, h in enumerate(history[1:], 1):
                n_good = sum(1 for r in h["results"] if r["ok"] and not r["issues"])
                with st.expander("Antes · %s · «%s» · %d ✅" % (h["time"], h["q"], n_good)):
                    if h["ctx"] != context:
                        st.caption("Sobre un paso anterior del análisis.")
                    _grid(h["results"], H.known_names(h["ctx"]), key="h%d" % i)


def _council_live(pending, providers, status, ss, history):
    """Pregunta en curso: módulos, barra de progreso por IA y cada respuesta en cuanto llega."""
    active = [p for p in providers if H.usable(p)]
    _modules(providers, status)
    st.markdown('<div class="hud-line">&gt; «%s» · leyendo tu análisis… (%d IA)</div>'
                % (html.escape(pending["q"]), len(active)), unsafe_allow_html=True)
    cols = st.columns(min(2, max(1, len(active))))
    slots = {p["id"]: cols[i % len(cols)].empty() for i, p in enumerate(active)}
    for p in active:
        slots[p["id"]].markdown(_thinking(p), unsafe_allow_html=True)
    known = H.known_names(pending["ctx"])
    results = []
    for r in H.ask_iter(providers, pending["q"], pending["ctx"]):
        results.append(r)
        with slots[r["id"]].container():
            _answer_card(r, known, key="live")
    order = {p["id"]: i for i, p in enumerate(providers)}
    results.sort(key=lambda r: order[r["id"]])
    history.insert(0, {"q": pending["q"], "ctx": pending["ctx"], "results": results, "time": time.strftime("%H:%M"),
                       "kind": pending.get("kind", "question")})
    del history[MAX_HISTORY:]
    cur = ss.get("ai_status") or {"fp": H.fingerprint(providers), "by_id": H.initial_status(providers)}
    ss["ai_status"] = {"fp": cur["fp"], "by_id": H.status_after_answers(cur["by_id"], results)}
    st.rerun()                                         # semáforo y consenso con todo ya llegado


# ---------------------------------------------------------------- «¿Listo para trabajar?» y órdenes

def _welcome_later():
    st.session_state["ai_welcome"] = "later"


@st.dialog("🤖 Asistente CMS-80", width="large", on_dismiss=_welcome_later)
def _welcome(names, items):
    """Ventana al entrar: las IA conectadas preguntan si empezamos. Nada se ejecuta sin un clic."""
    import flow
    todo = [it for it in items if it["estado"] != flow.OK]
    st.markdown("**Hola. Tengo %d IA conectada%s: %s.**" % (len(names), "" if len(names) == 1 else "s", ", ".join(names)))
    st.markdown("Antes de trabajar reviso todo el flujo con las reglas de la app (archivo, región, nulo, controles, "
                "difracción, otro filtro, control conocido) y les pido a las IA que **propongan órdenes**. "
                "Tú marcas cuáles se ejecutan: nada se hace sin tu clic.")
    if todo:
        st.markdown("Ahora mismo:\n" + "\n".join("- %s **%s**: %s" % (flow.ICON[it["estado"]], it["paso"], it["detalle"])
                                                for it in todo[:3]))
    st.markdown("**¿Listo para trabajar?**")
    c1, c2 = st.columns([3, 2])
    if c1.button("✅ Sí: revisa el flujo y propón órdenes", key="ai_welcome_yes", type="primary", use_container_width=True):
        st.session_state["ai_welcome"] = "yes"
        st.session_state["flow_review_request"] = True
        st.rerun()
    if c2.button("Ahora no", key="ai_welcome_no", use_container_width=True):
        _welcome_later()
        st.rerun()


def _latest_orders():
    """Órdenes de la última revisión con IA, si es sobre el estado actual."""
    ss = st.session_state
    h = next((h for h in ss.get("ai_history") or [] if h.get("kind") == "orders"), None)
    if not h or h["ctx"] != ss.get("ai_context_now"):
        return {}, (h["time"] if h else None)
    return H.consensus_orders(h["results"]), h["time"]


def render_orders(items, feasible, on_run):
    """Ventana «Órdenes de trabajo»: la revisión del flujo por reglas, las órdenes que proponen las reglas y
    las IA (con sus motivos y votos verificados) y la cola de ejecución. La persona marca y ejecuta."""
    import flow
    ss = st.session_state
    ai, ai_time = _latest_orders()
    rules = flow.suggested(items)
    cands = [a for a in dict.fromkeys(sorted(ai, key=lambda a: -ai[a]["votes"]) + rules) if a in feasible]
    queue, now = ss.get("order_queue") or [], ss.get("order_now")
    with st.container(key="win_orders"):
        theme.window_title("Órdenes de trabajo", "📋", "revisión del flujo: %s" % flow.summary(items))
        with st.expander("🧭 Revisión del flujo (reglas fijas de la app, no IA) · %s" % flow.summary(items),
                         expanded=not ai and any(it["estado"] != flow.OK for it in items)):
            for it in items:
                st.markdown("%s **%s** — %s" % (flow.ICON[it["estado"]], it["paso"], it["detalle"]))
        if now or queue:
            st.info("⏳ Ejecutando: **%s**%s" % (ACTION_LABELS.get(now, "—").lstrip("▶ "),
                    (" · después: " + ", ".join(ACTION_LABELS[a].lstrip("▶ ") for a in queue)) if queue else ""))
            if st.button("⏹ Parar después de esta orden", key="orders_stop"):
                ss["order_queue"] = []
                st.rerun()
        elif cands:
            st.markdown("**Órdenes propuestas** (marca las que quieras; se ejecutan en orden: cargar → analizar → "
                        "afinar → confirmar → control):")
            picked = []
            for a in cands:
                o = ai.get(a, {"votes": 0, "by": [], "reasons": []})
                rule = next((it for it in items if it.get("orden") == a and it["estado"] != flow.OK), None)
                label = ACTION_LABELS[a] + ("  · 🤖 %d IA" % o["votes"] if o["votes"] else "") + ("  · 🧠 regla" if rule else "")
                default = o["votes"] > 0 or (not ai and rule is not None and a == rules[0])
                if st.checkbox(label, value=default, key="order_pick_" + a):
                    picked.append(a)
                why = (["🧠 %s: %s" % (rule["paso"], rule["detalle"])] if rule else []) + \
                      ["🤖 %s: %s%s" % (name, reason or "(sin motivo)", "" if good else " ⚠️ no pasó la comprobación: no vota")
                       for name, reason, good in o["reasons"]]
                if why:
                    st.caption("  \n".join(why))
            if st.button("▶ Ejecutar las órdenes marcadas (%d)" % len(picked), key="orders_run", type="primary",
                         disabled=not picked, use_container_width=True):
                on_run(flow.in_pipeline_order(picked))
        else:
            st.caption("No hay órdenes pendientes que se puedan ejecutar ahora.")
        if ai_time:
            st.caption("🤖 Propuestas de las IA de las %s (las que no pasan la comprobación no votan)." % ai_time)
        if ss.get("ai_context_now") and not (now or queue):
            c1, c2 = st.columns([2, 3])
            if c1.button("🤖 Revisar el flujo con IA", key="orders_review"):
                ss["flow_review_request"] = True
                st.rerun()
            c2.checkbox("Revisar con IA al terminar las órdenes", value=True, key="auto_review_after")
        for line in (ss.get("order_log") or [])[-4:]:
            st.caption(line)
