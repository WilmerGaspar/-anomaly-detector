"""🤖 Centro de conexiones IA: varias IA gratuitas a la vez, cada una con su semáforo.

Proveedores con capa gratuita y API compatible con OpenAI (misma forma de llamada): NVIDIA
build, Groq, OpenRouter (solo modelos gratis: su enrutador «openrouter/free» y un DeepSeek «:free»
elegido de su lista pública) y Google Gemini (AI Studio). Cada uno se activa
poniendo su clave en Streamlit → Settings → Secrets; sin clave queda ⚪ y la app sigue igual.

Reglas para que las IA no decidan ni inventen:
- Solo reciben el resumen del análisis (números del JSON, comprobaciones del semáforo, tabla
  de hipótesis) y la pregunta. El semáforo y las medidas los calcula la app, no la IA.
- Cada respuesta se VERIFICA con código: los números que cita deben estar en los datos y no
  puede afirmar descubrimientos. Si falla, se marca ⚠️ y se dice por qué.
- Pueden PROPONER una acción de una lista fija; la persona la ejecuta con un clic.
Las condiciones gratuitas cambian (límites no siempre publicados): los modelos se pueden
cambiar en Secrets (<PROVEEDOR>_MODEL) sin tocar código.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata

PROVIDERS = [
    # Modelo y opción de pensamiento tomados del ejemplo oficial de build.nvidia.com (10-oct-2026).
    # Sin pensamiento: responde antes y no gasta los tokens razonando (las respuestas son cortas y
    # se comprueban contra los datos). El razonamiento, si llega, viene aparte en reasoning_content.
    {"id": "nvidia", "name": "NVIDIA", "base_url": "https://integrate.api.nvidia.com/v1",
     "key": "NVIDIA_API_KEY", "model_key": "NVIDIA_MODEL", "default_model": "nvidia/nemotron-3.5-lightning-30b-a3b",
     "signup": "https://build.nvidia.com", "extra": {"chat_template_kwargs": {"enable_thinking": False}}},
    {"id": "groq", "name": "Groq", "base_url": "https://api.groq.com/openai/v1",
     "key": "GROQ_API_KEY", "model_key": "GROQ_MODEL", "default_model": "openai/gpt-oss-120b",
     "signup": "https://console.groq.com", "extra": {"reasoning_effort": "low"}},
    # OpenRouter retira modelos «:free» sin aviso (10-oct-2026: gpt-oss-120b:free y deepseek-r1-0528:free
    # responden 404 «unavailable for free» y proponen la versión de pago). Se usa el MEJOR gratis del día
    # de su lista pública (ver best_free): su enrutador «openrouter/free» elige al azar y llegó a contestar
    # un modelo de 2,6B que se inventó datos. El enrutador queda de último recurso.
    {"id": "openrouter", "name": "OpenRouter", "base_url": "https://openrouter.ai/api/v1",
     "key": "OPENROUTER_API_KEY", "model_key": "OPENROUTER_MODEL", "default_model": "openrouter/free",
     "signup": "https://openrouter.ai/collections/free-models", "free_pick": "best"},
    # DeepSeek (R1 y sucesores, MIT): no caben en Streamlit Cloud, se usan gratis por OpenRouter con la
    # MISMA clave OPENROUTER_API_KEY. El modelo se elige solo de la lista pública de OpenRouter (precio 0
    # y «:free»; R1 primero si vuelve a ser gratis); el de reserva solo se usa si la lista no se puede leer.
    {"id": "deepseek", "name": "DeepSeek", "base_url": "https://openrouter.ai/api/v1",
     "key": "OPENROUTER_API_KEY", "model_key": "DEEPSEEK_MODEL", "default_model": "deepseek/deepseek-v4-flash:free",
     "signup": "https://openrouter.ai/models?q=deepseek", "free_family": "deepseek/"},
    {"id": "gemini", "name": "Gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
     "key": "GEMINI_API_KEY", "model_key": "GEMINI_MODEL", "default_model": "gemini-3.5-flash",
     "signup": "https://aistudio.google.com", "extra": {"reasoning_effort": "low"}},
]

NO_KEY, UNTESTED, CONNECTED, ERROR = "no_key", "untested", "connected", "error"
RESTING = "resting"           # con clave, pero el servicio no ofrece hoy ningún modelo gratis de esa IA: no es un fallo
LIGHT = {NO_KEY: "⚪", UNTESTED: "🟡", CONNECTED: "🟢", ERROR: "🔴", RESTING: "💤"}
STATE_TEXT = {NO_KEY: "sin clave", UNTESTED: "clave puesta, sin probar", CONNECTED: "conectada", ERROR: "error",
              RESTING: "en pausa: sin modelo gratis hoy"}

ACTIONS = {"hazlo_por_mi": "auto", "otro_archivo": "other_filter", "analizar": "auto_analyze",
           "mas_subrogados": "more_null", "control_conocido": "known_control"}


# ---------------------------------------------------------------- configuración

def configured(get_secret, get=None):
    """Proveedores con su clave y modelo resueltos. `get_secret(nombre)` -> str o None.
    Con `free_family` y sin modelo puesto a mano, el modelo sale de la lista de gratis de OpenRouter."""
    out = []
    for p in PROVIDERS:
        key = (get_secret(p["key"]) or "").strip()
        chosen = (get_secret(p["model_key"]) or "").strip()
        q = dict(p, api_key=key, model=chosen or p["default_model"])
        if p.get("free_pick") and key and not chosen:
            # El mejor gratis de hoy (no uno al azar): sin los de las otras IA con la misma clave.
            best = best_free(get=get, exclude=[x["free_family"] for x in PROVIDERS if x.get("free_family")])
            if best:
                cands = [e["id"] for e in best[:3]] + [p["default_model"]]      # el enrutador, de último recurso
                q.update(model=cands[0], candidates=cands, auto_model=True, size_b=best[0]["size_b"])
        elif p.get("free_family") and key and not chosen:
            cands = free_candidates(p["free_family"], get=get)
            if cands:
                q.update(model=cands[0], candidates=cands[:3], auto_model=True)
            elif cands is not None:                    # lista leída y ninguno gratis
                q["model"] = "(ninguno gratis ahora)"
                q["unavailable"] = ("OpenRouter no ofrece hoy ningún modelo %s gratis (solo de pago). No es un fallo "
                                    "tuyo ni de la clave: las demás IA siguen y la app vuelve a mirar la lista cada %d h"
                                    % (p["name"], FREE_LIST_TTL // 3600))
        out.append(q)
    return out


# ---------------------------------------------------------------- modelos gratis de OpenRouter

# Lista pública (sin clave): {"data": [{"id", "created", "pricing": {"prompt": "0", "completion": "0"}}…]}.
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
FREE_LIST_TTL = 6 * 3600          # se vuelve a leer cada 6 h (los gratis cambian a menudo)
FREE_LIST_RETRY = 600             # si falló, se reintenta a los 10 min
_FREE = {"t": 0.0, "ids": None, "entries": None, "bad": set()}


def _models_json(get, timeout=8):
    r = get(OPENROUTER_MODELS_URL, timeout=timeout)
    return r.json() if getattr(r, "status_code", 0) == 200 else None


# Tamaño (miles de millones de parámetros) según el nombre («gemma-4-31b», «gpt-oss-120b»; no «a3b», que son
# los activos) o, si no lo dice, según la descripción («284B total parameters», «2.6 billion parameters»).
_SIZE_ID = re.compile(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)b(?![a-z0-9])")
_SIZE_DESC = re.compile(r"(\d+(?:\.\d+)?)\s*(?:([bt])\b|(billion|trillion))[\s-]*(?:total\s+)?param", re.I)
MIN_GOOD_B = 20                   # por debajo, los modelos se inventan datos con facilidad (caso real: 2,6B)
_SKIP = re.compile(r"guard|embed|rerank|moderat", re.I)          # no son modelos para conversar


def model_size_b(model_id, description=""):
    """Parámetros en miles de millones que dice el modelo de sí mismo, o None si no lo dice."""
    name = str(model_id or "").lower().split("/")[-1].split(":")[0]
    found = [float(x) for x in _SIZE_ID.findall(name)]
    if found:
        return max(found)
    sizes = [float(n) * (1000.0 if (u or w or "").lower() in ("t", "trillion") else 1.0)
             for n, u, w in _SIZE_DESC.findall(str(description or ""))]
    return max(sizes) if sizes else None


def free_entries(data):
    """Modelos gratis de verdad: precio 0 de entrada y de salida y terminados en «:free». Más nuevos primero."""
    out = []
    for m in (data or {}).get("data") or []:
        mid, pr = str(m.get("id") or ""), m.get("pricing") or {}
        try:
            free = float(pr.get("prompt", 1)) == 0 and float(pr.get("completion", 1)) == 0
        except (TypeError, ValueError):
            free = False
        if free and mid.endswith(":free"):
            arch = m.get("architecture") or {}
            ins, outs = arch.get("input_modalities"), arch.get("output_modalities")
            out.append({"id": mid, "created": int(m.get("created") or 0),
                        "size_b": model_size_b(mid, m.get("description")),
                        "text": (not ins or "text" in ins) and (not outs or "text" in outs)})
    return sorted(out, key=lambda e: e["created"], reverse=True)


def free_ids(data):
    return [e["id"] for e in free_entries(data)]


def free_models(get=None, now=None):
    """Gratis de OpenRouter ahora (guardado 6 h para todas las sesiones); None si no se pudo leer."""
    now = time.time() if now is None else now
    ttl = FREE_LIST_TTL if _FREE["ids"] is not None else FREE_LIST_RETRY
    if _FREE["t"] and now - _FREE["t"] < ttl:
        return _FREE["ids"]
    if get is None:
        import requests
        get = requests.get
    try:
        data = _models_json(get)
        entries = free_entries(data) if isinstance(data, dict) and data.get("data") else None
    except Exception:                              # red, JSON roto…: se usa el modelo de reserva
        entries = None
    ids = [e["id"] for e in entries] if entries is not None else None
    _FREE.update(t=now, ids=ids, entries=entries)
    if ids is not None:
        _FREE["bad"] = set()                       # lista nueva: un modelo que volvió a ser gratis vale otra vez
    return ids


def _entries(get=None):
    ids = free_models(get)
    if ids is None:
        return None
    entries = _FREE.get("entries")
    if entries is None or [e["id"] for e in entries] != ids:          # lista puesta solo con nombres
        entries = [{"id": i, "created": 0, "size_b": model_size_b(i), "text": True} for i in ids]
    return entries


def best_free(get=None, exclude=()):
    """Los gratis de hoy, del más capaz al menos: primero los de ≥ 20B (más grande primero), luego los que no
    dicen su tamaño y al final los pequeños; en cada grupo, los de programación al final. None si no hay lista."""
    entries = _entries(get)
    if entries is None:
        return None

    def rank(e):
        size = e["size_b"]
        tier = 0 if (size or 0) >= MIN_GOOD_B else (1 if size is None else 2)
        return (tier, "code" in e["id"].lower(), -(size or 0), -e["created"])
    pool = [e for e in entries if e["text"] and e["id"] not in _FREE["bad"] and not _SKIP.search(e["id"])
            and not any(e["id"].startswith(f) for f in exclude)]
    return sorted(pool, key=rank)


def free_candidates(family, get=None):
    """Gratis de una familia («deepseek/»), R1 primero y luego del más nuevo; sin los que ya fallaron."""
    ids = free_models(get)
    if ids is None:
        return None
    fam = [i for i in ids if i.startswith(family) and i not in _FREE["bad"]]
    return sorted(fam, key=lambda i: 0 if "-r1" in i else 1)        # orden estable: dentro, más nuevo primero


def _model_gone(body):
    """El servicio dice que ese modelo ya no existe o ya no es gratis (no si es la privacidad)."""
    low = _provider_message(body).lower()
    return "data policy" not in low and any(k in low for k in MODEL_GONE)


def _try_next(code, body):
    """Con un modelo elegido solo, ¿probar el siguiente gratis? Si ya no existe o no es gratis, si lo prohíbe
    (403: p. ej. solo para agentes) o si su proveedor está saturado (429 «upstream»; el límite diario de la
    cuenta no, porque vale para todos). Devuelve (probar_otro, no_volver_a_usarlo)."""
    if _model_gone(body) or code == 403:
        return True, True
    if code == 429 and "upstream" in _provider_message(body).lower():
        return True, False
    return False, False


def usable(provider):
    """Tiene clave y un modelo gratis que usar: se prueba y se le pregunta."""
    return bool(provider.get("api_key")) and not provider.get("unavailable")


def initial_status(providers):
    def first(p):
        if not p["api_key"]:
            return {"state": NO_KEY, "detail": "", "model": p["model"]}
        if p.get("unavailable"):
            return {"state": RESTING, "detail": p["unavailable"], "model": p["model"]}
        return {"state": UNTESTED, "detail": "", "model": p["model"]}
    return {p["id"]: first(p) for p in providers}


# ---------------------------------------------------------------- llamadas

def _provider_message(body):
    """Mensaje de error del servicio (formato OpenAI/OpenRouter: {"error": {"message": …}}), en una
    línea y sin nada con forma de clave."""
    try:
        data = json.loads(body) if isinstance(body, str) else body
    except (TypeError, ValueError):
        data = None
    if isinstance(data, dict):                     # JSON: solo su mensaje de error, nunca el cuerpo entero
        err = data.get("error")
        msg = err.get("message") if isinstance(err, dict) else (err if isinstance(err, str) else data.get("message"))
    else:
        msg = body
    msg = re.sub(r"\s+", " ", str(msg or "")).strip()
    msg = re.sub(r"(nvapi-|sk-or-|gsk_|AIza)[\w-]+", "[clave]", msg)
    return msg[:180]


# Mensajes de «ese modelo ya no existe o ya no es gratis» (con un modelo elegido solo, se pasa al siguiente).
MODEL_GONE = ("unavailable for free", "paid version", "not a valid model", "model not found", "does not exist",
              "no endpoints found", "unknown model", "not found for api")
CHANGE_MODEL = "«Detalle de cada IA» → «Cambiar el modelo»"

# Qué hacer según el mensaje del servicio (se busca en minúsculas; el primero que encaja gana).
# OpenRouter devuelve 404 «No endpoints found matching your data policy» cuando su configuración de
# privacidad no permite los modelos gratuitos (sus proveedores pueden guardar las preguntas).
ERROR_ADVICE = [
    (("data policy", "privacy"),
     "OpenRouter bloquea los modelos gratis por tu configuración de privacidad. Arreglo: entra en "
     "openrouter.ai/settings/privacy y permite los modelos gratuitos (pueden guardar las preguntas; la app "
     "solo envía números públicos del telescopio)"),
    # 10-oct-2026: «This model is unavailable for free. The paid version is available now - use this slug
    # instead: openai/gpt-oss-120b». El nombre que propone es DE PAGO: no se cambia a él nunca.
    (("unavailable for free", "paid version"),
     "OpenRouter retiró la versión gratis de este modelo. El nombre que propone (sin «:free») es de pago y "
     "necesita saldo: no lo pongas. Arreglo: en %s deja el campo vacío (la app usa uno gratis de hoy) o escribe "
     "openrouter/free" % CHANGE_MODEL),
    (("no auth credentials", "user not found", "invalid api key", "incorrect api key", "api key not valid",
      "invalid_api_key", "unauthorized", "authentication"),
     "clave no válida: vuelve a copiarla entera desde la web del servicio"),
    (("not a valid model", "model not found", "does not exist", "no endpoints found", "unknown model",
      "not found for api"),
     "ese modelo ya no está disponible: cámbialo en %s (vacío = el de la app) o en Secrets" % CHANGE_MODEL),
    (("rate limit", "rate-limit", "per-day", "quota", "too many requests"),
     "límite gratuito alcanzado: espera unos minutos (o hasta mañana si es el límite diario)"),
    (("credits", "insufficient", "payment"),
     "la cuenta no tiene saldo para este modelo: usa un modelo gratuito (los que terminan en :free)"),
]


def _error_text(code, body=""):
    """Motivo del error en claro, qué hacer y el mensaje original del servicio."""
    msg = _provider_message(body)
    low = msg.lower()
    advice = next((a for keys, a in ERROR_ADVICE if any(k in low for k in keys)), None)
    if advice is None:
        if code in (401, 403):
            advice = "clave no válida o sin permiso"
        elif code == 404:
            advice = "modelo no encontrado: cámbialo en %s" % CHANGE_MODEL
        elif code == 429:
            advice = "límite gratuito alcanzado: espera un minuto"
        elif code == 402:
            advice = "la cuenta no tiene saldo para este modelo: usa un modelo gratuito"
        elif code and code >= 500:
            advice = "el servicio falla ahora mismo: prueba más tarde"
        else:
            advice = "respuesta inesperada"
    out = "%s (%s)" % (advice, code)
    if msg and msg.lower() not in advice.lower():
        out += " · mensaje del servicio: «%s»" % msg
    return out


# Modelos de razonamiento tipo DeepSeek-R1. Recomendaciones de su repositorio: temperatura 0.5-0.7
# (0.6; con valores bajos se repite sin fin) y sin mensaje de sistema (todo en el mensaje del
# usuario). Piensan antes de responder: más tokens y más tiempo de espera.
REASONING = {"temperature": 0.6, "min_tokens": 6000, "timeout": 150}


def is_reasoning(provider):
    return bool(provider.get("reasoning")) or "deepseek-r1" in str(provider.get("model", "")).lower()


def _as_user_only(messages):
    """Sin mensaje de sistema: sus instrucciones van delante del primer mensaje del usuario."""
    system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
    rest = [dict(m) for m in messages if m.get("role") != "system"]
    if system and rest and rest[0].get("role") == "user":
        rest[0]["content"] = system + "\n\n" + rest[0]["content"]
    elif system:
        rest.insert(0, {"role": "user", "content": system})
    return rest


def chat(provider, messages, max_tokens=1200, timeout=60, post=None):
    """Una petición de chat. Devuelve dict(ok, text, detail, latency_ms, model, served_model).
    `model` es el pedido; `served_model`, el que contestó según el servicio (con openrouter/free es el
    modelo gratis que eligió el enrutador)."""
    if post is None:
        import requests
        post = requests.post
    model = provider.get("model")
    if not provider.get("api_key"):
        return {"ok": False, "text": "", "detail": "sin clave", "latency_ms": None, "model": model}
    if provider.get("unavailable"):
        return {"ok": False, "text": "", "detail": provider["unavailable"], "latency_ms": None, "model": model}
    url = provider["base_url"].rstrip("/") + "/chat/completions"
    headers = {"Authorization": "Bearer %s" % provider["api_key"], "Content-Type": "application/json"}
    t0 = time.monotonic()
    # Modelo elegido de la lista de gratis: si el servicio dice que ya no existe o ya no es gratis, se
    # prueba el siguiente gratis (nunca la versión de pago que propone) y no se vuelve a usar.
    for i, model in enumerate(provider.get("candidates") or [model]):
        p = dict(provider, model=model)
        msgs, temperature, tokens, wait = messages, 0.2, int(max_tokens), timeout
        if is_reasoning(p):
            msgs, temperature = _as_user_only(messages), REASONING["temperature"]
            if max_tokens > 64:                  # la prueba de conexión sigue siendo mínima
                tokens, wait = max(tokens, REASONING["min_tokens"]), max(timeout, REASONING["timeout"])
        payload = {"model": model, "messages": msgs, "max_tokens": tokens, "temperature": temperature}
        # Razonamiento corto (más rápido y deja tokens para la respuesta), solo con el modelo por
        # defecto: otro modelo puesto en Secrets podría no aceptar la opción.
        extra = provider.get("extra") if model == provider.get("default_model") else None
        try:
            r = post(url, headers=headers, json=dict(payload, **(extra or {})), timeout=wait)
            if extra and getattr(r, "status_code", 0) == 400:
                r = post(url, headers=headers, json=payload, timeout=wait)    # sin la opción
        except Exception as exc:                   # red, tiempo agotado, DNS…
            name = type(exc).__name__
            detail = "no responde (tiempo agotado)" if "Timeout" in name else "sin conexión con el servicio (%s)" % name
            return {"ok": False, "text": "", "detail": detail, "latency_ms": None, "model": model}
        code, body = getattr(r, "status_code", 0), getattr(r, "text", "")
        if code != 200 and provider.get("auto_model"):
            again, retired = _try_next(code, body)
            if retired:
                _FREE["bad"].add(model)
            if again and i + 1 < len(provider["candidates"]):
                continue
        break
    ms = int(1000 * (time.monotonic() - t0))
    if code != 200:
        return {"ok": False, "text": "", "detail": _error_text(code, body), "latency_ms": ms, "model": model}
    try:
        data = r.json()
        text = data["choices"][0]["message"].get("content") or ""
    except Exception:
        return {"ok": False, "text": "", "detail": "respuesta con formato desconocido", "latency_ms": ms, "model": model}
    served = data.get("model") if isinstance(data.get("model"), str) else None
    return {"ok": True, "text": strip_reasoning(text), "detail": "", "latency_ms": ms, "model": model,
            "served_model": served}


def strip_reasoning(text):
    """Algunos modelos de razonamiento escriben su borrador entre <think>…</think> dentro de la
    respuesta. Se quita; si el borrador no se cerró (se acabaron los tokens), no queda respuesta."""
    text = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.S | re.I)
    if re.search(r"<think>", text, flags=re.I):
        text = re.split(r"<think>", text, flags=re.I)[0]
    return text.strip()


def ping(provider, post=None):
    """Prueba de conexión mínima (pocos tokens). Estado para el semáforo."""
    if not provider.get("api_key"):
        return {"state": NO_KEY, "detail": "", "model": provider["model"]}
    if provider.get("unavailable"):                # no se llama: hoy no hay modelo gratis
        return {"state": RESTING, "detail": provider["unavailable"], "model": provider["model"]}
    r = chat(provider, [{"role": "user", "content": "Responde solo: OK"}], max_tokens=16, timeout=15, post=post)
    # Con HTTP 200 está conectada aunque un modelo de razonamiento no llegue a escribir "OK".
    ok = r["ok"]
    return {"state": CONNECTED if ok else ERROR, "detail": "" if ok else r["detail"],
            "model": r.get("model") or provider["model"], "latency_ms": r["latency_ms"]}


def ping_all(providers, post=None):
    """Prueba todas las conexiones en paralelo. {id: estado}."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(providers) or 1) as ex:
        states = list(ex.map(lambda p: ping(p, post=post), providers))
    return {p["id"]: s for p, s in zip(providers, states)}


def fingerprint(providers):
    """Qué proveedores tienen clave y con qué modelo (sin la clave): si cambia en Secrets, se vuelve a probar."""
    return tuple((p["id"], bool(p["api_key"]), p["model"]) for p in providers)


def status_after_answers(status, results):
    """Una consulta también dice si la conexión funciona: el semáforo se actualiza con ella."""
    status = dict(status)
    for r in results:
        if r["ok"]:
            status[r["id"]] = {"state": CONNECTED, "detail": "", "model": r["model"], "latency_ms": r["latency_ms"]}
        else:
            status[r["id"]] = {"state": ERROR, "detail": r["detail"], "model": r["model"], "latency_ms": r["latency_ms"]}
    return status


def light_line(providers, status):
    """Una línea con el semáforo de cada IA: «🟢 NVIDIA · ⚪ Groq · …»."""
    return " · ".join("%s %s" % (LIGHT[status.get(p["id"], {}).get("state", NO_KEY)], p["name"]) for p in providers)


def status_detail(provider, s):
    """Texto de una fila del semáforo, sin la clave."""
    state = s.get("state", NO_KEY)
    parts = ["%s **%s**" % (LIGHT[state], provider["name"]), STATE_TEXT[state]]
    if state != NO_KEY:
        parts.append("`%s`" % s.get("model", provider["model"]))
        if provider.get("auto_model") and provider.get("free_pick"):
            size = provider.get("size_b")
            parts.append("el mejor gratis de hoy" + (" (≈%gB parámetros)" % size if size else ""))
        elif provider.get("auto_model"):
            parts.append("gratis hoy, elegido solo")
        elif provider["model"] == "openrouter/free":
            parts.append("elige un modelo gratis en cada pregunta")
    if state == CONNECTED and s.get("latency_ms") is not None:
        parts.append("%.1f s" % (s["latency_ms"] / 1000.0))
    if state in (ERROR, RESTING) and s.get("detail"):
        parts.append(s["detail"])
    return " · ".join(parts)


def _messages(question, context):
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "DATOS:\n%s\n\nPREGUNTA: %s" % (context, question)}]


def _result(p, a, question, context):
    if a["ok"] and not a["text"]:
        # HTTP 200 sin texto: el modelo gastó los tokens razonando y no llegó a responder.
        a = dict(a, ok=False, detail="respuesta vacía (el modelo no terminó de responder)")
    text, orders = split_orders(a["text"]) if a["ok"] else ("", [])
    text, action = split_action(text) if a["ok"] else ("", None)
    issues = verify(text, context + "\n" + question) if a["ok"] else []
    return {"id": p["id"], "name": p["name"], "model": a.get("model") or p["model"], "ok": a["ok"], "text": text,
            "action": action, "orders": orders, "issues": issues, "detail": a["detail"], "latency_ms": a["latency_ms"],
            "served_model": a.get("served_model")}


def ask_iter(providers, question, context, post=None):
    """La misma pregunta a todas las IA con clave, en paralelo; entrega cada resultado (ya
    verificado) en cuanto llega, para que la pantalla lo muestre sin esperar a la más lenta."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    active = [p for p in providers if usable(p)]
    if not active:
        return
    messages = _messages(question, context)
    with ThreadPoolExecutor(max_workers=len(active)) as ex:
        futures = {ex.submit(chat, p, messages, ANSWER_TOKENS, 60, post): p for p in active}
        for fut in as_completed(futures):
            yield _result(futures[fut], fut.result(), question, context)


def ask_all(providers, question, context, post=None):
    """Todas las respuestas, en el orden de la lista de proveedores."""
    order = {p["id"]: i for i, p in enumerate(providers)}
    return sorted(ask_iter(providers, question, context, post=post), key=lambda r: order[r["id"]])


# Preguntas de un clic en el panel: cubren lo que más se pregunta tras un análisis.
QUICK_QUESTIONS = [
    ("📋 Explícame el resultado", "Explícame el resultado y cuál es el siguiente paso."),
    ("🧪 ¿Qué hipótesis y qué prueba la decide?",
     "De la tabla de hipótesis, ¿cuáles son compatibles y qué prueba concreta de la tabla decidiría entre ellas?"),
    ("⚠️ ¿Qué NO puedo afirmar?", "¿Qué NO se puede afirmar con estos datos y por qué?"),
    ("🔭 ¿Qué hago ahora?", "¿Cuál es la siguiente acción más útil y por qué?"),
]


# ---------------------------------------------------------------- claves pegadas en la app

KEY_PREFIXES = (("nvapi-", "NVIDIA_API_KEY"), ("gsk_", "GROQ_API_KEY"), ("sk-or-", "OPENROUTER_API_KEY"),
                ("AIza", "GEMINI_API_KEY"))


def detect_key_name(key):
    """Qué proveedor es una clave por cómo empieza (NVIDIA nvapi-, Groq gsk_, OpenRouter sk-or-,
    Gemini AIza). None si no se reconoce: entonces la persona elige el proveedor."""
    k = str(key or "").strip()
    for prefix, name in KEY_PREFIXES:
        if k.startswith(prefix):
            return name
    return None


def providers_for_key(name):
    """Nombres de las IA que se encienden con esa clave (la de OpenRouter enciende dos)."""
    return [p["name"] for p in PROVIDERS if p["key"] == name]


# ---------------------------------------------------------------- contexto e instrucciones

# Los modelos de razonamiento (gpt-oss, Nemotron, Gemini) gastan parte de estos tokens pensando.
ANSWER_TOKENS = 2048
DEFAULT_QUESTION = "Explícame el resultado y cuál es el siguiente paso."

SYSTEM_PROMPT = (
    "Eres el intérprete de CMS-80, una app que analiza imágenes de telescopios (FITS de Hubble y JWST). "
    "Hablas en español claro a una persona sin formación técnica.\n"
    "REGLAS (obligatorias; si no puedes cumplir una, dilo):\n"
    "1. Usa SOLO los DATOS. Para lo que no esté en ellos responde exactamente: «No lo sé con estos datos».\n"
    "2. Detrás de cada número o afirmación sobre el análisis, pon entre corchetes de qué dato sale, con su nombre en "
    "los DATOS. Ejemplo: β = 2.6 [beta_espectro]; 🟠 [semaforo].\n"
    "3. Copia los números tal cual. No inventes números, objetos, distancias, estrellas ni referencias: solo puedes "
    "citar las referencias que aparecen en la tabla de HIPOTESIS.\n"
    "4. El semáforo y la tabla de hipótesis los calcula la app con pruebas: no los cambies ni los contradigas. "
    "Una hipótesis «compatible» no está demostrada.\n"
    "5. Nunca digas que algo es un descubrimiento, física nueva o algo seguro: como mucho, una pregunta por comprobar.\n"
    "6. Puedes explicar conceptos generales (qué es β, qué es un filtro), marcándolos como (concepto general).\n"
    "FORMATO (máximo 220 palabras, con estos cuatro títulos en negrita):\n"
    "**Qué muestra:** …\n**Qué significa:** …\n**Qué NO se puede afirmar:** …\n**Siguiente prueba:** …\n"
    "Termina con una línea «ACCIÓN: X», donde X es una de ACCIONES_DISPONIBLES (si no aplica, «ninguna»)."
)


def _intermittency(card):
    """Curtosis de incrementos a 1 px frente al 95 % del nulo (saltos bruscos de brillo)."""
    sc = (card.get("analytics") or {}).get("scales") or {}
    fl, q95 = sc.get("flatness") or [], sc.get("flatness_null_q95") or []
    return {"curtosis_1px": fl[0], "nulo_95": q95[0]} if fl and q95 else None


def _hot_tiles(card, n=3):
    """Las zonas del mapa local con más señal, en píxeles de la imagen (x, y) y su z."""
    lm = (card.get("analytics") or {}).get("local_map") or {}
    z, tile = lm.get("z"), lm.get("tile_px")
    crop = ((card.get("analysis") or {}).get("crop")) or {}
    if not z or not tile:
        return None
    cells = [(float(v), i, j) for i, row in enumerate(z) for j, v in enumerate(row) if v is not None and v == v]
    out = []
    for v, i, j in sorted(cells, reverse=True)[:n]:
        x0, y0 = crop.get("x0", 0) + j * tile[1], crop.get("y0", 0) + i * tile[0]
        out.append({"x": [x0, x0 + tile[1]], "y": [y0, y0 + tile[0]], "z": round(v, 1)})
    return out


def build_context(card, log=None, step=None, actions=(), gate=None, flow=None):
    """Resumen compacto (JSON) de lo que la IA puede usar. Nada más sale de la app.
    `gate`: semáforo que muestra la app ahora (con la bitácora como referencia puede pasar de
    🟢 a 🟣); si no se da, el del JSON. `flow`: revisión del flujo por reglas (flow.review)."""
    ctx = {"ACCIONES_DISPONIBLES": ["ninguna"] + [k for k, v in ACTIONS.items() if v in actions]}
    if step:
        ctx["PASO_ACTUAL"] = {"titulo": step.get("title"), "texto": step.get("text"), "pasos": step.get("steps")}
    if flow:
        back = {v: k for k, v in ACTIONS.items()}
        ctx["REVISION_FLUJO"] = [{"paso": it["paso"], "estado": it["estado"], "detalle": it["detalle"],
                                  "orden": back.get(it.get("orden"))} for it in flow]
    if card:
        src, gate = card.get("source") or {}, gate or card.get("discovery_gate") or {}
        st_ = card.get("structure_test") or {}
        fdr = st_.get("fdr") or {}
        d = card.get("descriptors") or {}
        fit = ((card.get("analytics") or {}).get("spectrum") or {}).get("fit") or {}
        ctx["ANALISIS"] = {
            "archivo": src.get("filename"), "instrumento": src.get("instrument"), "filtro": src.get("filter"),
            "objeto": src.get("target"),
            "semaforo": {"nivel": gate.get("level"), "titulo": gate.get("title"), "significado": gate.get("meaning"),
                         "comprobaciones": [{"prueba": c.get("check"), "superada": c.get("ok"), "detalle": c.get("detail")}
                                            for c in gate.get("checks") or []]},
            "prueba_frente_al_nulo": {"p": st_.get("p_value"), "z": st_.get("z_score"), "pasan_fdr": fdr.get("n_passed"),
                                      "de": fdr.get("n_tested"), "descriptores_que_pasan": fdr.get("passed_descriptors")},
            "control_estrellas": {k: (card.get("discovery_controls") or {}).get("point_sources_masked", {}).get(k)
                                  for k in ("n_masked", "masked_fraction", "valid", "n_passed")},
            "repeticion_otra_semilla": (card.get("discovery_controls") or {}).get("replicate"),
            "difraccion": {"beta_sin_corregir": fit.get("beta_raw"), "corregida": fit.get("psf_corrected"),
                           "beta_no_medible": fit.get("psf_limited"),
                           "lambda_um": ((card.get("analysis") or {}).get("psf") or {}).get("lambda_um")},
            "direccion": {"estructuras_en_el_cielo_PA": (d.get("anisotropy") or {}).get("structure_pa_deg"),
                          "estructuras_en_la_imagen": (d.get("anisotropy") or {}).get("structure_direction_degrees"),
                          "coherencia": (d.get("anisotropy") or {}).get("orientation_coherence"),
                          "coherencia_nulo_95": (d.get("anisotropy") or {}).get("orientation_null_q95"),
                          "p_direccion": (d.get("anisotropy") or {}).get("orientation_p")},
            "centro_RA_Dec": [src.get("center_ra_deg"), src.get("center_dec_deg")],
            "intermitencia": _intermittency(card),
            "zonas_mas_fuertes": _hot_tiles(card),
            "medidas": {"beta_espectro": fit.get("beta"), "r2_ajuste": fit.get("r2"),
                        "lacunaridad": (d.get("fractal_base") or {}).get("lacunarity"),
                        "multifractalidad": (d.get("fractal_base") or {}).get("multifractality_index"),
                        "betti_0": (d.get("persistent_homology") or {}).get("betti_0"),
                        "betti_1": (d.get("persistent_homology") or {}).get("betti_1"),
                        "anisotropia": (d.get("anisotropy") or {}).get("anisotropy_index"),
                        "exceso_crestas": (d.get("ridges") or {}).get("filament_excess"),
                        "fuentes_puntuales": (card.get("morphology") or {}).get("n_point_sources")},
        }
        hyp = card.get("hypotheses") or {}
        if hyp.get("active"):
            ctx["HIPOTESIS"] = {"clasificacion": hyp.get("title"),
                                "tabla": [{"mecanismo": r["mechanism"], "estado": r["status"], "por_que": r["why"],
                                           "pregunta": r["question"], "referencia": r["refs"]} for r in hyp.get("rows", [])]}
        nov = (gate or {}).get("novelty") or {}
        if nov.get("available"):
            ctx["ANALISIS"]["novedad_frente_a_tus_analisis"] = {"max_z": nov.get("max_abs_z"),
                                                                "referencias": nov.get("n_references")}
    if log:
        ctx["BITACORA"] = [{"objeto": e.get("target"), "archivo": e.get("file"), "filtro": e.get("filter"),
                            "semaforo": e.get("level")} for e in log[-10:]]
    return json.dumps(ctx, ensure_ascii=False, default=str)


def _paths(obj, prefix=""):
    """Todos los nombres de datos del contexto, con y sin la ruta («beta_espectro», «medidas.beta_espectro»…)."""
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            full = (prefix + "." + k) if prefix else k
            parts = full.split(".")
            out.update(".".join(parts[i:]) for i in range(len(parts)))
            out |= _paths(v, full)
    elif isinstance(obj, list):
        for v in obj:
            out |= _paths(v, prefix)
    return out


# ---------------------------------------------------------------- verificación

_NUM = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)?")
# «Apellido 1999», «Apellido et al. 1999», «Apellido & Otro 1999», «Apellido (1999)».
_REF = re.compile(r"\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)(?:\s+(?:et\s+al\.?|y|&|and)\s*(?:[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)?)?"
                  r"[\s,]*\(?((?:19|20)\d{2})\)?")
# Citas del dato de origen: [beta_espectro], [semaforo], [medidas.beta_espectro]…
_CITE = re.compile(r"\[([A-Za-z_][\w.]*)\](?!\()")          # no los enlaces [texto](url)
OVERCLAIMS = ("descubrimiento confirmado", "hemos descubierto", "has descubierto", "se ha descubierto",
              "nueva fisica", "fisica nueva", "demuestra que", "sin duda", "con total certeza", "es seguro que")


def _norm(t):
    t = unicodedata.normalize("NFD", str(t).lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def _numbers(text):
    out = []
    for m in _NUM.findall(str(text)):
        s = m.replace(",", ".")
        try:
            out.append((float(s), len(s.split(".")[1]) if "." in s else 0))
        except ValueError:
            pass
    return out


def known_names(allowed_text):
    """Nombres de datos que se pueden citar (normalizados), del JSON de la primera línea; None si no hay JSON."""
    try:
        return ({_norm(k) for k in _paths(json.loads(allowed_text.split("\n", 1)[0]))}
                if str(allowed_text).lstrip().startswith("{") else None)
    except ValueError:
        return None


def cite_ok(name, known):
    """¿Existe en los datos el dato citado entre corchetes? (sin datos con los que comparar: sí)."""
    return known is None or _norm(str(name).strip()) in known


def verify(answer, allowed_text):
    """Problemas de una respuesta: números que no salen de los datos (o de la pregunta) y
    afirmaciones de más. Enteros pequeños (0-10) se permiten: enumeraciones y conteos."""
    issues = []
    allowed = [v for v, _ in _numbers(allowed_text)]
    bad = []
    for v, dec in _numbers(answer):
        if dec == 0 and abs(v) <= 10:
            continue
        tol = 0.5 * 10 ** (-dec) + 1e-9
        ok = any(abs(v - a) <= tol or abs(v / 100.0 - a) <= tol / 100.0 or abs(v - 100 * a) <= tol for a in allowed)
        if not ok:
            bad.append(("%g" % v).replace(".", ","))
    if bad:
        issues.append("cita números que no están en tus datos: %s" % ", ".join(sorted(set(bad))[:5]))
    low = _norm(answer)
    claims = [c for c in OVERCLAIMS if c in low]
    if claims:
        issues.append("afirma de más («%s»)" % claims[0])
    allowed_norm = _norm(allowed_text)
    refs = [m.group(1) for m in _REF.finditer(str(answer)) if _norm(m.group(1)) not in allowed_norm]
    if refs:
        issues.append("cita referencias que no están en tus datos: %s" % ", ".join(sorted(set(refs))[:3]))
    known_norm = known_names(allowed_text)
    if known_norm is not None:
        bad_cites = [c for c in _CITE.findall(str(answer)) if _norm(c.strip()) not in known_norm]
        if bad_cites:
            issues.append("cita datos que no existen: %s" % ", ".join("[%s]" % c for c in sorted(set(bad_cites))[:3]))
    return issues


# Pregunta de la ventana «¿Listo para trabajar?» y del botón «Revisar el flujo con IA».
REVIEW_QUESTION = (
    "Revisa el flujo de trabajo como un revisor exigente: usa REVISION_FLUJO (comprobaciones hechas por la app con "
    "reglas fijas) y el ANALISIS. Di qué está bien, qué falta o falla y en qué orden conviene seguir. Al final, además "
    "de la línea «ACCIÓN:», escribe hasta 3 líneas «ORDEN: clave — motivo [dato]», con claves de "
    "ACCIONES_DISPONIBLES, de la más importante a la menos.")

_ORDER_LINE = re.compile(r"\s*\**\s*orden\s*\**\s*:\s*\**\s*([a-z_áéíóú]+)\**\s*(?:[—–:|-]+\s*(.*))?$", re.I)


def split_orders(text):
    """Separa las líneas «ORDEN: clave — motivo». Devuelve (texto, [(acción de la guía, motivo)]); solo
    claves de la lista cerrada, sin repetir, como mucho 3."""
    orders, keep = [], []
    for line in str(text).splitlines():
        m = _ORDER_LINE.match(line)
        if m:
            a = ACTIONS.get(_norm(m.group(1)).strip())
            if a and a not in [o for o, _ in orders] and len(orders) < 3:
                orders.append((a, (m.group(2) or "").strip()))
            continue
        keep.append(line)
    return "\n".join(keep).strip(), orders


def split_action(text):
    """Separa la línea «ACCIÓN: X» del texto. Devuelve (texto, clave de acción de la guía o None)."""
    action = None
    keep = []
    for line in str(text).splitlines():
        m = re.match(r"\s*\**\s*acci[oó]n\s*\**\s*:\s*\**\s*([a-z_áéíóú]+)", line, flags=re.I)
        if m:
            action = ACTIONS.get(_norm(m.group(1)).strip())
            continue
        keep.append(line)
    return "\n".join(keep).strip(), action


def same_model_key(r):
    """Quién respondió de verdad: el modelo que dice el servicio (sin «:free»), o la IA si no lo dice."""
    served = str(r.get("served_model") or "").strip().lower()
    return served[:-5] if served.endswith(":free") else (served or "id:" + str(r["id"]))


def consensus(results):
    """Acción propuesta por la mayoría de las IA que respondieron sin problemas de verificación. Si dos
    respuestas vienen del mismo modelo (openrouter/free puede elegir el mismo que otra IA), vota una vez."""
    votes, seen = [], set()
    for r in results:
        if r["ok"] and not r["issues"] and r["action"] and same_model_key(r) not in seen:
            seen.add(same_model_key(r))
            votes.append(r["action"])
    if not votes:
        return None, 0, len([r for r in results if r["ok"]])
    best = max(set(votes), key=votes.count)
    return best, votes.count(best), len([r for r in results if r["ok"]])


def consensus_orders(results):
    """Órdenes propuestas por las IA: {acción: {"votes", "by", "reasons"}}. Solo votan las respuestas que
    pasan la comprobación y el mismo modelo vota una vez; las demás se ven, marcadas, pero no cuentan."""
    out, seen = {}, {}
    for r in results:
        if not r.get("ok"):
            continue
        for a, why in (r.get("orders") or ([(r["action"], "")] if r.get("action") else [])):
            o = out.setdefault(a, {"votes": 0, "by": [], "reasons": []})
            o["reasons"].append((r["name"], why, not r["issues"]))
            who = same_model_key(r)
            if not r["issues"] and who not in seen.setdefault(a, set()):
                seen[a].add(who)
                o["votes"] += 1
                o["by"].append(r["name"])
    return out
