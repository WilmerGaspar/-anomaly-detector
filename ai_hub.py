"""🤖 Centro de conexiones IA: varias IA gratuitas a la vez, cada una con su semáforo.

Proveedores con capa gratuita y API compatible con OpenAI (misma forma de llamada): NVIDIA
build, Groq, OpenRouter (modelos ":free") y Google Gemini (AI Studio). Cada uno se activa
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
    {"id": "openrouter", "name": "OpenRouter", "base_url": "https://openrouter.ai/api/v1",
     "key": "OPENROUTER_API_KEY", "model_key": "OPENROUTER_MODEL", "default_model": "openai/gpt-oss-120b:free",
     "signup": "https://openrouter.ai"},
    # DeepSeek-R1 (MIT, 671B parámetros, 37B activos): no cabe en Streamlit Cloud, se usa gratis por
    # OpenRouter con la MISMA clave OPENROUTER_API_KEY. Es un modelo de razonamiento: ver REASONING.
    {"id": "deepseek", "name": "DeepSeek R1", "base_url": "https://openrouter.ai/api/v1",
     "key": "OPENROUTER_API_KEY", "model_key": "DEEPSEEK_MODEL", "default_model": "deepseek/deepseek-r1-0528:free",
     "signup": "https://openrouter.ai/deepseek/deepseek-r1-0528:free", "reasoning": True},
    {"id": "gemini", "name": "Gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
     "key": "GEMINI_API_KEY", "model_key": "GEMINI_MODEL", "default_model": "gemini-3.5-flash",
     "signup": "https://aistudio.google.com", "extra": {"reasoning_effort": "low"}},
]

NO_KEY, UNTESTED, CONNECTED, ERROR = "no_key", "untested", "connected", "error"
LIGHT = {NO_KEY: "⚪", UNTESTED: "🟡", CONNECTED: "🟢", ERROR: "🔴"}
STATE_TEXT = {NO_KEY: "sin clave", UNTESTED: "clave puesta, sin probar", CONNECTED: "conectada", ERROR: "error"}

ACTIONS = {"hazlo_por_mi": "auto", "otro_archivo": "other_filter", "analizar": "auto_analyze"}


# ---------------------------------------------------------------- configuración

def configured(get_secret):
    """Proveedores con su clave y modelo resueltos. `get_secret(nombre)` -> str o None."""
    out = []
    for p in PROVIDERS:
        key = (get_secret(p["key"]) or "").strip()
        model = (get_secret(p["model_key"]) or "").strip() or p["default_model"]
        out.append(dict(p, api_key=key, model=model))
    return out


def initial_status(providers):
    return {p["id"]: {"state": UNTESTED if p["api_key"] else NO_KEY, "detail": "", "model": p["model"]}
            for p in providers}


# ---------------------------------------------------------------- llamadas

def _error_text(code, body=""):
    body = str(body or "")[:200]
    if code in (401, 403):
        return "clave no válida o sin permiso (%d)" % code
    if code == 404:
        return "modelo no encontrado (%d): cambia el modelo en Secrets" % code
    if code == 429:
        return "límite gratuito alcanzado (429): espera un minuto"
    if code and code >= 500:
        return "el servicio falla ahora mismo (%d)" % code
    return "respuesta inesperada (%s) %s" % (code, body)


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
    """Una petición de chat. Devuelve dict(ok, text, detail, latency_ms)."""
    if post is None:
        import requests
        post = requests.post
    if not provider.get("api_key"):
        return {"ok": False, "text": "", "detail": "sin clave", "latency_ms": None}
    url = provider["base_url"].rstrip("/") + "/chat/completions"
    temperature = 0.2
    if is_reasoning(provider):
        messages = _as_user_only(messages)
        temperature = REASONING["temperature"]
        if max_tokens > 64:                      # la prueba de conexión sigue siendo mínima
            max_tokens = max(int(max_tokens), REASONING["min_tokens"])
            timeout = max(timeout, REASONING["timeout"])
    payload = {"model": provider["model"], "messages": messages, "max_tokens": int(max_tokens), "temperature": temperature}
    # Razonamiento corto (más rápido y deja tokens para la respuesta), solo con el modelo por
    # defecto: otro modelo puesto en Secrets podría no aceptar la opción.
    extra = provider.get("extra") if provider["model"] == provider.get("default_model") else None
    headers = {"Authorization": "Bearer %s" % provider["api_key"], "Content-Type": "application/json"}
    t0 = time.monotonic()
    try:
        r = post(url, headers=headers, json=dict(payload, **(extra or {})), timeout=timeout)
        if extra and getattr(r, "status_code", 0) == 400:
            r = post(url, headers=headers, json=payload, timeout=timeout)    # sin la opción
    except Exception as exc:                       # red, tiempo agotado, DNS…
        name = type(exc).__name__
        detail = "no responde (tiempo agotado)" if "Timeout" in name else "sin conexión con el servicio (%s)" % name
        return {"ok": False, "text": "", "detail": detail, "latency_ms": None}
    ms = int(1000 * (time.monotonic() - t0))
    if getattr(r, "status_code", 0) != 200:
        return {"ok": False, "text": "", "detail": _error_text(getattr(r, "status_code", 0), getattr(r, "text", "")),
                "latency_ms": ms}
    try:
        text = r.json()["choices"][0]["message"].get("content") or ""
    except Exception:
        return {"ok": False, "text": "", "detail": "respuesta con formato desconocido", "latency_ms": ms}
    return {"ok": True, "text": strip_reasoning(text), "detail": "", "latency_ms": ms}


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
    r = chat(provider, [{"role": "user", "content": "Responde solo: OK"}], max_tokens=16, timeout=15, post=post)
    # Con HTTP 200 está conectada aunque un modelo de razonamiento no llegue a escribir "OK".
    ok = r["ok"]
    return {"state": CONNECTED if ok else ERROR, "detail": "" if ok else r["detail"], "model": provider["model"],
            "latency_ms": r["latency_ms"]}


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
        parts.append("`%s`" % provider["model"])
    if state == CONNECTED and s.get("latency_ms") is not None:
        parts.append("%.1f s" % (s["latency_ms"] / 1000.0))
    if state == ERROR and s.get("detail"):
        parts.append(s["detail"])
    return " · ".join(parts)


def _messages(question, context):
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "DATOS:\n%s\n\nPREGUNTA: %s" % (context, question)}]


def _result(p, a, question, context):
    if a["ok"] and not a["text"]:
        # HTTP 200 sin texto: el modelo gastó los tokens razonando y no llegó a responder.
        a = dict(a, ok=False, detail="respuesta vacía (el modelo no terminó de responder)")
    text, action = split_action(a["text"]) if a["ok"] else ("", None)
    issues = verify(text, context + "\n" + question) if a["ok"] else []
    return {"id": p["id"], "name": p["name"], "model": p["model"], "ok": a["ok"], "text": text,
            "action": action, "issues": issues, "detail": a["detail"], "latency_ms": a["latency_ms"]}


def ask_iter(providers, question, context, post=None):
    """La misma pregunta a todas las IA con clave, en paralelo; entrega cada resultado (ya
    verificado) en cuanto llega, para que la pantalla lo muestre sin esperar a la más lenta."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    active = [p for p in providers if p.get("api_key")]
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


def build_context(card, log=None, step=None, actions=(), gate=None):
    """Resumen compacto (JSON) de lo que la IA puede usar. Nada más sale de la app.
    `gate`: semáforo que muestra la app ahora (con la bitácora como referencia puede pasar de
    🟢 a 🟣); si no se da, el del JSON."""
    ctx = {"ACCIONES_DISPONIBLES": ["ninguna"] + [k for k, v in ACTIONS.items() if v in actions]}
    if step:
        ctx["PASO_ACTUAL"] = {"titulo": step.get("title"), "texto": step.get("text"), "pasos": step.get("steps")}
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
    try:
        known = _paths(json.loads(allowed_text.split("\n", 1)[0])) if allowed_text.lstrip().startswith("{") else None
    except ValueError:
        known = None
    if known is not None:
        known_norm = {_norm(k) for k in known}
        bad_cites = [c for c in _CITE.findall(str(answer)) if _norm(c.strip()) not in known_norm]
        if bad_cites:
            issues.append("cita datos que no existen: %s" % ", ".join("[%s]" % c for c in sorted(set(bad_cites))[:3]))
    return issues


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


def consensus(results):
    """Acción propuesta por la mayoría de las IA que respondieron sin problemas de verificación."""
    votes = [r["action"] for r in results if r["ok"] and not r["issues"] and r["action"]]
    if not votes:
        return None, 0, len([r for r in results if r["ok"]])
    best = max(set(votes), key=votes.count)
    return best, votes.count(best), len([r for r in results if r["ok"]])
