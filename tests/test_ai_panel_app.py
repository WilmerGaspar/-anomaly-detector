"""🤖 IA conectadas en la app real (streamlit.testing): semáforo por IA, consejo verificado y
propuesta que la persona acepta con un clic. Las IA y MAST se simulan: no hay red ni claves reales."""
import json
from pathlib import Path

import requests

from test_autopilot_app import fake_mast  # noqa: F401  (fixture: MAST simulado)

ROOT = Path(__file__).resolve().parents[1]
KEYS = {"NVIDIA_API_KEY": "nvapi-SECRETO-1", "GROQ_API_KEY": "gsk-SECRETO-2", "OPENROUTER_API_KEY": "sk-or-SECRETO-3"}


class _Resp:
    def __init__(self, code, content=""):
        self.status_code = code
        self._body = {"choices": [{"message": {"content": content}}]}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


def _fake_ai(monkeypatch):
    calls = []

    def post(url, headers=None, json=None, timeout=None, **kw):
        calls.append({"url": url, "json": json})
        if "openrouter" in url:
            return _Resp(401)                                    # clave mal copiada
        if json["max_tokens"] <= 16:
            return _Resp(200, "OK")                               # prueba de conexión
        data = json["messages"][1]["content"]
        assert "ACCIONES_DISPONIBLES" in data and "hazlo_por_mi" in data
        if "nvidia" in url:
            return _Resp(200, "Primero hay que buscar y cargar un archivo.\nACCIÓN: hazlo_por_mi")
        return _Resp(200, "El índice vale 7.77, sin duda.\nACCIÓN: hazlo_por_mi")      # Groq inventa

    monkeypatch.setattr(requests, "post", post)
    return calls


def _texts(at):
    out = [m.value for m in at.main.markdown] + [c.value for c in at.main.caption]
    return " ".join(str(t) for t in out)


def test_without_keys_everything_is_white_and_nothing_is_sent(monkeypatch):
    for k in list(KEYS) + ["GEMINI_API_KEY"]:
        monkeypatch.delenv(k, raising=False)
    calls = _fake_ai(monkeypatch)
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    text = _texts(at)
    assert all("⚪ %s" % n in text for n in ("NVIDIA", "Groq", "OpenRouter", "DeepSeek", "Gemini"))
    assert not [b for b in at.main.button if b.key == "ai_ask"] and calls == []


def test_lights_council_and_accepted_proposal(monkeypatch, fake_mast):  # noqa: F811
    for k, v in KEYS.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    calls = _fake_ai(monkeypatch)
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    # Al abrir se prueba cada conexión una vez: dos 🟢, la clave mala 🔴 con el motivo (también
    # en DeepSeek, que usa la misma clave de OpenRouter), Gemini ⚪.
    text = _texts(at)
    assert all(c in text for c in ("🟢 NVIDIA", "🟢 Groq", "🔴 OpenRouter", "🔴 DeepSeek", "⚪ Gemini"))
    assert "clave no válida o sin permiso (401)" in text
    assert len(calls) == 4
    at.run()
    assert len(calls) == 4                                       # no se vuelve a probar en cada recarga

    next(b for b in at.main.button if b.key == "ai_ask").click().run()
    assert not at.exception, [e.value for e in at.exception]
    text = _texts(at)
    assert "✅ pasa la comprobación" in text and "Primero hay que buscar" in text
    assert "⚠️ cita números que no están en tus datos: 7,77" in text and "afirma de más" in text
    assert "clave no válida o sin permiso (401)" in text and "🧭" in text          # tarjeta en 🔴 y consenso
    assert "SECRETO" not in text + " ".join(str(m.value) for m in at.markdown)   # la clave nunca se ve
    # Solo cuenta la respuesta que pasa la comprobación: 1 de 2.
    do = [b for b in at.main.button if b.key == "ai_do"]
    assert do and "(1 de 2 IA)" in do[0].label and "Hazlo por mí" in do[0].label

    do[0].click().run()
    for _ in range(4):
        if at.session_state["guide_log"] if "guide_log" in at.session_state else None:
            break
        at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert fake_mast["download"] == ["mast:ok"] and len(at.session_state["guide_log"]) == 1
    # Tras el análisis las respuestas son de un paso anterior: no se ofrece repetir la acción.
    assert "paso anterior" in _texts(at) and not [b for b in at.main.button if b.key == "ai_do"]

    # Con un resultado, el resumen que reciben las IA lleva las medidas reales del JSON.
    next(b for b in at.main.button if b.key == "ai_ask").click().run()
    last_nv = [c for c in calls if "nvidia" in c["url"]][-1]          # los hilos no llegan en orden
    ctx = json.loads(last_nv["json"]["messages"][1]["content"].split("DATOS:\n", 1)[1].split("\n\nPREGUNTA:")[0])
    a = ctx["ANALISIS"]
    assert a["archivo"] == "jw_prueba_i2d.fits" and a["filtro"] == "F200W" and a["semaforo"]["nivel"]
    assert a["prueba_frente_al_nulo"]["p"] is not None and a["prueba_frente_al_nulo"]["de"] == 4
    m = a["medidas"]
    for k in ("beta_espectro", "lacunaridad", "multifractalidad", "betti_0", "anisotropia", "exceso_crestas"):
        assert m[k] is not None, k
    assert ctx["ACCIONES_DISPONIBLES"][0] == "ninguna" and len(ctx["BITACORA"]) == 1


def test_paste_a_key_in_the_app(monkeypatch):
    """Sin Secrets: se pega la clave en la app, se reconoce por su prefijo (nvapi- → NVIDIA), se
    prueba la conexión con ella y nunca aparece en pantalla."""
    for k in list(KEYS) + ["GEMINI_API_KEY"]:
        monkeypatch.delenv(k, raising=False)
    seen = []

    def post(url, headers=None, json=None, timeout=None, **kw):
        seen.append((url, headers["Authorization"]))
        return _Resp(200, "OK")

    monkeypatch.setattr(requests, "post", post)
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    box = next(t for t in at.text_input if t.label == "Pega aquí tu clave API")
    box.input("nvapi-SECRETO-PEGADO")
    next(b for b in at.button if "Conectar" in str(b.label)).click().run()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["ai_session_keys"] == {"NVIDIA_API_KEY": "nvapi-SECRETO-PEGADO"}
    assert seen and seen[0][0].startswith("https://integrate.api.nvidia.com") and seen[0][1].endswith("SECRETO-PEGADO")
    text = _texts(at) + " ".join(str(m.value) for m in at.markdown)
    assert "🟢 NVIDIA" in text and "SECRETO" not in text
    assert [b for b in at.main.button if b.key == "ai_ask"]                      # ya se puede preguntar


def test_red_light_shows_the_reason_and_model_can_change_in_the_app(monkeypatch):
    """OpenRouter con la privacidad cerrada: 404 «data policy». El motivo y el arreglo se ven bajo el
    semáforo (una línea para OpenRouter y DeepSeek) y el modelo se puede cambiar sin Secrets."""
    for k in list(KEYS) + ["GEMINI_API_KEY"]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-SECRETO-4")
    seen = []

    class _Err:
        status_code = 404
        text = json.dumps({"error": {"message": "No endpoints found matching your data policy (Free model "
                                                "publication). Configure: https://openrouter.ai/settings/privacy"}})

    def post(url, headers=None, json=None, timeout=None, **kw):
        seen.append(json["model"])
        return _Err()

    monkeypatch.setattr(requests, "post", post)
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    warns = [w.value for w in at.warning]
    assert any("OpenRouter y DeepSeek" in w and "openrouter.ai/settings/privacy" in w for w in warns), warns
    box = next(t for t in at.text_input if t.label == "DeepSeek")
    box.input("deepseek/deepseek-r1:free")
    next(b for b in at.button if "Usar estos modelos" in str(b.label)).click().run()
    at.run()
    assert "deepseek/deepseek-r1:free" in seen                      # la prueba usa el modelo nuevo
    assert "SECRETO" not in " ".join(str(m.value) for m in at.markdown) + " ".join(warns)


def test_retired_free_models_screenshot_case(monkeypatch):
    """Caso real (captura del 10-oct-2026): con los modelos de antes, OpenRouter y DeepSeek R1 en 🔴 por
    «This model is unavailable for free. The paid version is available now - use this slug instead: …».
    Ahora: OpenRouter usa openrouter/free, DeepSeek uno gratis de la lista (si el primero ya no lo es,
    el siguiente) y nunca se pide la versión de pago."""
    for k in list(KEYS) + ["GEMINI_API_KEY", "OPENROUTER_MODEL", "DEEPSEEK_MODEL"]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-SECRETO-5")
    seen = []

    def gone(paid):
        r = _Resp(404)
        r.text = json.dumps({"error": {"message": "This model is unavailable for free. The paid version is available "
                                                  "now - use this slug instead: %s" % paid, "code": 404}})
        return r

    def post(url, headers=None, json=None, timeout=None, **kw):
        seen.append(json["model"])
        if json["model"] in ("openai/gpt-oss-120b:free", "deepseek/deepseek-v4-flash-0731:free"):
            return gone(json["model"][:-5])
        return _Resp(200, "OK")

    monkeypatch.setattr(requests, "post", post)
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    text = _texts(at) + " ".join(str(m.value) for m in at.markdown)
    assert "🟢 OpenRouter" in text and "🟢 DeepSeek" in text and not at.warning
    assert "openrouter/free" in seen and "deepseek/deepseek-v4-flash:free" in seen
    assert not [m for m in seen if not (m.endswith(":free") or m == "openrouter/free")]     # nada de pago
    # Si la persona pone a mano el modelo retirado, el 🔴 explica qué hacer (no «ve a Secrets»).
    box = next(t for t in at.text_input if t.label == "OpenRouter")
    box.input("openai/gpt-oss-120b:free")
    next(b for b in at.button if "Usar estos modelos" in str(b.label)).click().run()
    at.run()
    warns = " ".join(w.value for w in at.warning)
    assert "retiró la versión gratis" in warns and "Cambiar el modelo" in warns and "openrouter/free" in warns
    assert "SECRETO" not in warns + " ".join(str(m.value) for m in at.markdown)


def test_no_free_deepseek_today_is_a_calm_pause_not_a_red_error(monkeypatch):
    """Caso real (captura del 10-oct-2026, tras el cambio): OpenRouter 🟢 y DeepSeek 🔴 con un recuadro de
    aviso aunque no había nada que arreglar (OpenRouter retiró todos los DeepSeek gratis). Ahora: 💤 y una
    línea tranquila; no se prueba ni se pregunta a DeepSeek y el botón cuenta solo las IA disponibles."""
    import ai_hub
    from conftest import SAMPLE_MODELS
    for k in list(KEYS) + ["GEMINI_API_KEY", "OPENROUTER_MODEL", "DEEPSEEK_MODEL"]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-SECRETO-6")
    no_deepseek = {"data": [m for m in SAMPLE_MODELS["data"] if not m["id"].startswith("deepseek/")]}
    monkeypatch.setattr(ai_hub, "_models_json", lambda get, timeout=8: no_deepseek)
    seen = []

    def post(url, headers=None, json=None, timeout=None, **kw):
        seen.append(json["model"])
        return _Resp(200, "OK" if json["max_tokens"] <= 16 else "Hay que analizar.\nACCIÓN: hazlo_por_mi")

    monkeypatch.setattr(requests, "post", post)
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    text = _texts(at)
    assert "🟢 OpenRouter" in text and "💤 DeepSeek" in text and "🔴 DeepSeek" not in text and not at.warning
    assert "No es un fallo" in text and seen == ["openrouter/free"]             # solo se prueba OpenRouter
    ask = next(b for b in at.main.button if b.key == "ai_ask")
    assert "(1)" in ask.label
    ask.click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert seen.count("openrouter/free") == 2 and len(seen) == 2                # DeepSeek no se consulta
    assert "🔴 **DeepSeek**" not in " ".join(str(m.value) for m in at.markdown)
