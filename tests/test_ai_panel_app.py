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
    assert "⚪ NVIDIA · ⚪ Groq · ⚪ OpenRouter · ⚪ Gemini" in text
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
    # Al abrir se prueba cada conexión una vez: dos 🟢, la clave mala 🔴 con el motivo, Gemini ⚪.
    text = _texts(at)
    assert "🟢 NVIDIA · 🟢 Groq · 🔴 OpenRouter · ⚪ Gemini" in text
    assert "clave no válida o sin permiso (401)" in text
    assert len(calls) == 3
    at.run()
    assert len(calls) == 3                                       # no se vuelve a probar en cada recarga

    next(b for b in at.main.button if b.key == "ai_ask").click().run()
    assert not at.exception, [e.value for e in at.exception]
    text = _texts(at)
    assert "✅ pasa la comprobación" in text and "Primero hay que buscar" in text
    assert "⚠️ cita números que no están en tus datos: 7,77" in text and "afirma de más" in text
    assert "🔴 OpenRouter: clave no válida" in text
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
    ctx = json.loads(calls[-1]["json"]["messages"][1]["content"].split("DATOS:\n", 1)[1].split("\n\nPREGUNTA:")[0])
    a = ctx["ANALISIS"]
    assert a["archivo"] == "jw_prueba_i2d.fits" and a["filtro"] == "F200W" and a["semaforo"]["nivel"]
    assert a["prueba_frente_al_nulo"]["p"] is not None and a["prueba_frente_al_nulo"]["de"] == 4
    m = a["medidas"]
    for k in ("beta_espectro", "lacunaridad", "multifractalidad", "betti_0", "anisotropia", "exceso_crestas"):
        assert m[k] is not None, k
    assert ctx["ACCIONES_DISPONIBLES"][0] == "ninguna" and len(ctx["BITACORA"]) == 1
