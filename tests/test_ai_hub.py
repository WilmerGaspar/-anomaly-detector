"""🤖 Centro de conexiones IA (ai_hub.py): semáforo por IA, consejo en paralelo y verificación,
sin red (las peticiones HTTP se simulan)."""
import json

import pytest

import ai_hub as H


class _Resp:
    def __init__(self, code=200, content="OK", body=None):
        self.status_code = code
        self._body = body if body is not None else {"choices": [{"message": {"content": content}}]}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


def _providers(**keys):
    return H.configured(lambda name: keys.get(name))


def test_configured_reads_keys_and_models_without_exposing_them():
    ps = _providers(NVIDIA_API_KEY=" nvapi-SECRETO ", GROQ_MODEL="llama-x")
    by = {p["id"]: p for p in ps}
    assert by["nvidia"]["api_key"] == "nvapi-SECRETO" and by["nvidia"]["model"] == H.PROVIDERS[0]["default_model"]
    assert by["groq"]["api_key"] == "" and by["groq"]["model"] == "llama-x"
    st = H.initial_status(ps)
    assert st["nvidia"]["state"] == H.UNTESTED and st["groq"]["state"] == H.NO_KEY
    line = H.light_line(ps, st)
    assert "🟡 NVIDIA" in line and "⚪ Groq" in line and "⚪ Gemini" in line
    shown = line + " ".join(H.status_detail(p, st[p["id"]]) for p in ps)
    assert "SECRETO" not in shown                            # la clave nunca se muestra


def test_chat_sends_openai_compatible_request():
    seen = {}

    def post(url, headers, json, timeout):
        seen.update(url=url, headers=headers, json=json)
        return _Resp(content="<think>borrador</think>Hola")

    p = _providers(GROQ_API_KEY="gsk_x")[1]
    r = H.chat(p, [{"role": "user", "content": "hola"}], post=post)
    assert r["ok"] and r["text"] == "Hola"                    # el borrador del razonamiento se quita
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer gsk_x" and seen["json"]["model"] == p["model"]


@pytest.mark.parametrize("code,needle", [(401, "clave no válida"), (403, "clave no válida"), (404, "modelo"),
                                         (429, "límite gratuito"), (503, "falla")])
def test_http_errors_become_red_with_reason(code, needle):
    p = _providers(NVIDIA_API_KEY="k")[0]
    s = H.ping(p, post=lambda *a, **k: _Resp(code, body={"error": "x"}))
    assert s["state"] == H.ERROR and needle in s["detail"]


def test_network_failures_and_odd_answers():
    p = _providers(NVIDIA_API_KEY="k")[0]

    class ReadTimeout(Exception):
        pass

    def boom(*a, **k):
        raise ReadTimeout()

    assert "tiempo agotado" in H.ping(p, post=boom)["detail"]
    assert "sin conexión" in H.ping(p, post=lambda *a, **k: (_ for _ in ()).throw(OSError()))["detail"]
    assert "formato" in H.chat(p, [], post=lambda *a, **k: _Resp(body={"raro": 1}))["detail"]
    # HTTP 200 sin texto (modelo de razonamiento con pocos tokens): la conexión funciona.
    assert H.ping(p, post=lambda *a, **k: _Resp(content=None))["state"] == H.CONNECTED
    assert H.strip_reasoning("<think>sin cerrar…") == ""
    assert H.ping(_providers()[0])["state"] == H.NO_KEY


def test_ping_all_gives_one_light_per_provider():
    ps = [p for p in _providers(NVIDIA_API_KEY="a", GROQ_API_KEY="b") if p["api_key"]]

    def post(url, **k):
        return _Resp(200) if "nvidia" in url else _Resp(401)

    st = H.ping_all(ps, post=post)
    assert st["nvidia"]["state"] == H.CONNECTED and st["groq"]["state"] == H.ERROR


DATA = json.dumps({"ANALISIS": {"medidas": {"beta_espectro": 2.84731, "lacunaridad": 0.1234,
                                            "anisotropia": 0.42}, "prueba": {"p": 0.01, "z": 3.456}}})


@pytest.mark.parametrize("answer", ["β vale 2,85 y la lacunaridad 0.12.", "z = 3.46 con p = 0.01",
                                    "La anisotropía es del 42 %.", "Hay 3 pasos: 1, 2 y 3."])
def test_verify_accepts_numbers_from_the_data(answer):
    assert H.verify(answer, DATA) == []


def test_verify_flags_invented_numbers_and_overclaims():
    issues = H.verify("β = 3.67, como dice Smith 2019.", DATA)
    assert len(issues) == 1 and "3,67" in issues[0] and "2019" in issues[0]
    assert any("afirma de más" in i for i in H.verify("Hemos descubierto física nueva.", DATA))
    assert any("afirma de más" in i for i in H.verify("Esto DEMUESTRA QUE hay turbulencia", DATA))


@pytest.mark.parametrize("line,action", [("ACCIÓN: analizar", "auto_analyze"), ("**Acción:** otro_archivo", "other_filter"),
                                         ("accion: hazlo_por_mi", "auto"), ("ACCIÓN: ninguna", None),
                                         ("ACCIÓN: borrar_todo", None)])
def test_split_action_only_maps_known_actions(line, action):
    text, a = H.split_action("Explicación.\n" + line)
    assert text == "Explicación." and a == action


def _res(name, action, ok=True, issues=()):
    return {"id": name, "name": name, "model": "m", "ok": ok, "text": "t", "action": action, "issues": list(issues),
            "detail": "", "latency_ms": 1}


def test_consensus_counts_only_verified_answers():
    rs = [_res("a", "auto"), _res("b", "auto"), _res("c", "auto_analyze"),
          _res("d", "auto_analyze", issues=["x"]), _res("e", "auto_analyze", issues=["x"]), _res("f", None, ok=False)]
    assert H.consensus(rs) == ("auto", 2, 5)
    assert H.consensus([_res("a", None)]) == (None, 0, 1)


def test_ask_all_in_parallel_with_verification():
    ps = _providers(NVIDIA_API_KEY="a", GROQ_API_KEY="b", OPENROUTER_API_KEY="c")
    sent = []

    def post(url, headers, json, timeout):
        sent.append(json)
        if "nvidia" in url:
            return _Resp(content="β = 2,85: pendiente medida.\nACCIÓN: analizar")
        if "groq" in url:
            return _Resp(content="β = 9.99, seguro.\nACCIÓN: analizar")
        return _Resp(content="")                              # vacía: no cuenta como respuesta

    out = {r["id"]: r for r in H.ask_all(ps, "¿Qué es β?", DATA, post=post)}
    # Gemini sin clave no se consulta; DeepSeek R1 usa la clave de OpenRouter.
    assert set(out) == {"nvidia", "groq", "openrouter", "deepseek"}
    assert out["nvidia"]["ok"] and out["nvidia"]["issues"] == [] and out["nvidia"]["action"] == "auto_analyze"
    assert "ACCIÓN" not in out["nvidia"]["text"]
    assert out["groq"]["issues"] and not out["openrouter"]["ok"] and "vacía" in out["openrouter"]["detail"]
    plain = [m for m in sent if len(m["messages"]) == 2]          # los que aceptan mensaje de sistema
    assert len(plain) == 3 and all(m["messages"][0]["content"] == H.SYSTEM_PROMPT for m in plain)
    assert all("PREGUNTA: ¿Qué es β?" in m["messages"][-1]["content"] for m in sent)
    status = H.status_after_answers(H.initial_status(ps), list(out.values()))
    assert status["nvidia"]["state"] == H.CONNECTED and status["openrouter"]["state"] == H.ERROR
    assert status["gemini"]["state"] == H.NO_KEY
    assert H.ask_all(_providers(), "x", DATA) == []


def test_context_lists_only_available_actions_and_live_gate():
    card = {"source": {"filename": "f_i2d.fits", "filter": "F200W"},
            "discovery_gate": {"level": "robust", "title": "Estructura robusta", "checks": []},
            "structure_test": {"p_value": 0.01, "z_score": 3.2, "fdr": {"n_passed": 2, "n_tested": 4}},
            "hypotheses": {"active": False}}
    ctx = json.loads(H.build_context(card, log=[{"file": str(i)} for i in range(15)],
                                     step={"title": "t", "text": "x", "steps": []}, actions=("other_filter",),
                                     gate={"level": "pioneer", "title": "Atípico", "checks": []}))
    assert ctx["ACCIONES_DISPONIBLES"] == ["ninguna", "otro_archivo"]
    assert ctx["ANALISIS"]["semaforo"]["nivel"] == "pioneer" and ctx["ANALISIS"]["prueba_frente_al_nulo"]["p"] == 0.01
    assert "HIPOTESIS" not in ctx and len(ctx["BITACORA"]) == 10
    assert json.loads(H.build_context(None))["ACCIONES_DISPONIBLES"] == ["ninguna"]


def test_low_reasoning_only_for_default_model_and_retried_without_it():
    sent = []

    def post(url, headers, json, timeout):
        sent.append(json)
        return _Resp(400, body={"error": "unknown"}) if "reasoning_effort" in json else _Resp(content="hola")

    groq = _providers(GROQ_API_KEY="k")[1]
    r = H.chat(groq, [], post=post)
    assert r["ok"] and [("reasoning_effort" in j) for j in sent] == [True, False]   # rechazada -> sin ella
    sent.clear()
    H.chat(_providers(GROQ_API_KEY="k", GROQ_MODEL="otro-modelo")[1], [], post=post)
    assert len(sent) == 1 and "reasoning_effort" not in sent[0]                       # modelo cambiado: no se envía
    sent.clear()
    H.chat(_providers(NVIDIA_API_KEY="k")[0], [], post=post)
    assert len(sent) == 1 and "reasoning_effort" not in sent[0]


def test_nvidia_uses_the_official_model_without_thinking():
    """Ejemplo de build.nvidia.com: nemotron-3.5-lightning-30b-a3b; el pensamiento se apaga con
    chat_template_kwargs.enable_thinking (si el servicio no lo acepta, se repite sin la opción)."""
    sent = []

    def post(url, headers, json, timeout):
        sent.append(json)
        return _Resp(content="hola")

    nv = _providers(NVIDIA_API_KEY="nvapi-x")[0]
    assert nv["model"] == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert H.chat(nv, [{"role": "user", "content": "x"}], post=post)["ok"]
    assert sent[0]["model"] == nv["model"] and sent[0]["chat_template_kwargs"] == {"enable_thinking": False}
    sent.clear()
    H.chat(_providers(NVIDIA_API_KEY="k", NVIDIA_MODEL="otro/modelo")[0], [], post=post)
    assert "chat_template_kwargs" not in sent[0]                      # otro modelo: sin opciones extra


def test_deepseek_r1_by_openrouter_follows_its_usage_rules():
    """DeepSeek-R1 (671B) no cabe en Streamlit Cloud: se usa por OpenRouter con la misma clave.
    Su repositorio pide temperatura 0.5-0.7 y sin mensaje de sistema; razona antes de responder."""
    sent = []

    def post(url, headers, json, timeout):
        sent.append((url, json, timeout))
        return _Resp(content="<think>razono…</think>β vale 2,85.\nACCIÓN: ninguna")

    ps = {p["id"]: p for p in _providers(OPENROUTER_API_KEY="sk-or-x", GROQ_API_KEY="g")}
    ds = ps["deepseek"]
    assert ds["api_key"] == ps["openrouter"]["api_key"] == "sk-or-x"          # una clave, dos IA
    assert ds["model"] == "deepseek/deepseek-r1-0528:free" and H.is_reasoning(ds)
    out = {r["id"]: r for r in H.ask_all([ds], "¿β?", DATA, post=post)}
    url, js, timeout = sent[0]
    assert url.startswith("https://openrouter.ai/api/v1") and js["temperature"] == 0.6
    assert [m["role"] for m in js["messages"]] == ["user"]                    # sin mensaje de sistema
    assert js["messages"][0]["content"].startswith(H.SYSTEM_PROMPT) and "PREGUNTA: ¿β?" in js["messages"][0]["content"]
    assert js["max_tokens"] >= 6000 and timeout >= 150
    assert out["deepseek"]["text"] == "β vale 2,85." and out["deepseek"]["issues"] == []   # sin el borrador
    sent.clear()
    assert H.ping(ds, post=post)["state"] == H.CONNECTED and sent[0][1]["max_tokens"] == 16   # prueba mínima
    sent.clear()
    H.chat(ps["groq"], [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], post=post)
    assert sent[0][1]["temperature"] == 0.2 and len(sent[0][1]["messages"]) == 2              # los demás, igual
