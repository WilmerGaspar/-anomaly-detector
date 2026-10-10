"""🤖 Centro de conexiones IA (ai_hub.py): semáforo por IA, consejo en paralelo y verificación,
sin red (las peticiones HTTP se simulan)."""
import json
import time

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
    assert len(issues) == 2 and "3,67" in issues[0] and "2019" in issues[0]
    assert "referencias" in issues[1] and "Smith" in issues[1]                 # referencia inventada
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
    # Gemini sin clave no se consulta; DeepSeek usa la clave de OpenRouter.
    assert set(out) == {"nvidia", "groq", "openrouter", "deepseek"}
    assert out["nvidia"]["ok"] and out["nvidia"]["issues"] == [] and out["nvidia"]["action"] == "auto_analyze"
    assert "ACCIÓN" not in out["nvidia"]["text"]
    assert out["groq"]["issues"] and not out["openrouter"]["ok"] and "vacía" in out["openrouter"]["detail"]
    plain = [m for m in sent if len(m["messages"]) == 2]          # sin R1 todos aceptan mensaje de sistema
    assert len(plain) == 4 and all(m["messages"][0]["content"] == H.SYSTEM_PROMPT for m in plain)
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
    Su repositorio pide temperatura 0.5-0.7 y sin mensaje de sistema; razona antes de responder.
    Desde oct-2026 R1 ya no es gratis por defecto: se pone a mano (DEEPSEEK_MODEL) si vuelve a serlo."""
    sent = []

    def post(url, headers, json, timeout):
        sent.append((url, json, timeout))
        return _Resp(content="<think>razono…</think>β vale 2,85.\nACCIÓN: ninguna")

    ps = {p["id"]: p for p in _providers(OPENROUTER_API_KEY="sk-or-x", GROQ_API_KEY="g",
                                         DEEPSEEK_MODEL="deepseek/deepseek-r1:free")}
    ds = ps["deepseek"]
    assert ds["api_key"] == ps["openrouter"]["api_key"] == "sk-or-x"          # una clave, dos IA
    assert ds["model"] == "deepseek/deepseek-r1:free" and H.is_reasoning(ds) and not ds.get("auto_model")
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


# ---------------------------------------------------------------- solo modelos gratis (OpenRouter)

def test_free_list_keeps_only_zero_price_free_ids_newest_first():
    from conftest import SAMPLE_MODELS
    ids = H.free_ids(SAMPLE_MODELS)
    assert ids == ["google/gemma-4-31b-it:free", "deepseek/deepseek-v4-flash-0731:free", "deepseek/deepseek-v4-flash:free"]
    # De pago (sin «:free»), enrutadores con precio -1 y precios raros: fuera.
    odd = {"data": [{"id": "x/y:free", "pricing": {"prompt": "0.001", "completion": "0"}},
                    {"id": "x/z:free", "pricing": {"prompt": None, "completion": "0"}}, {"id": "x/w:free"}]}
    assert H.free_ids(odd) == [] and H.free_ids(None) == [] and H.free_ids({}) == []
    # Si R1 vuelve a ser gratis, va primero.
    back = {"data": SAMPLE_MODELS["data"] + [{"id": "deepseek/deepseek-r1-0528:free", "created": 1,
                                              "pricing": {"prompt": "0", "completion": "0"}}]}
    H._FREE.update(t=time.time(), ids=H.free_ids(back))
    assert H.free_candidates("deepseek/")[:2] == ["deepseek/deepseek-r1-0528:free", "deepseek/deepseek-v4-flash-0731:free"]


def test_openrouter_defaults_are_free_and_never_the_paid_slug():
    """Caso real (10-oct-2026): «This model is unavailable for free. The paid version is available now - use
    this slug instead: openai/gpt-oss-120b» (y lo mismo con deepseek-r1-0528). La app solo usa gratis."""
    ps = {p["id"]: p for p in _providers(OPENROUTER_API_KEY="sk-or-x")}
    assert ps["openrouter"]["model"] == "openrouter/free"                        # enrutador oficial gratis
    ds = ps["deepseek"]
    assert ds["auto_model"] and ds["model"] == "deepseek/deepseek-v4-flash-0731:free" and not H.is_reasoning(ds)
    assert ds["candidates"] == ["deepseek/deepseek-v4-flash-0731:free", "deepseek/deepseek-v4-flash:free"]
    assert all(c.endswith(":free") for c in ds["candidates"])
    # Sin clave de OpenRouter no se lee la lista.
    H._FREE.update(t=0.0, ids=None)
    assert not _providers(NVIDIA_API_KEY="k")[3].get("auto_model") and H._FREE["t"] == 0.0


def test_retired_free_model_falls_to_the_next_free_one():
    retired = json.dumps({"error": {"message": "This model is unavailable for free. The paid version is available now "
                                               "- use this slug instead: deepseek/deepseek-v4-flash", "code": 404}})
    sent = []

    class _Gone:
        status_code, text = 404, retired

    def post(url, headers, json, timeout):
        sent.append(json["model"])
        if json["model"] == "deepseek/deepseek-v4-flash-0731:free":
            return _Gone()
        return _Resp(body={"model": json["model"], "choices": [{"message": {"content": "OK"}}]})

    ds = _providers(OPENROUTER_API_KEY="sk-or-x")[3]
    s = H.ping(ds, post=post)
    assert s["state"] == H.CONNECTED and s["model"] == "deepseek/deepseek-v4-flash:free"
    assert sent == ["deepseek/deepseek-v4-flash-0731:free", "deepseek/deepseek-v4-flash:free"]
    assert "deepseek/deepseek-v4-flash" not in sent                              # nunca la de pago
    # Ya no se vuelve a elegir el retirado.
    assert _providers(OPENROUTER_API_KEY="sk-or-x")[3]["model"] == "deepseek/deepseek-v4-flash:free"
    # Si fallan todos, 🔴 con el motivo claro.
    sent.clear()
    assert H.ping(dict(ds, candidates=["a/b:free"]), post=lambda *a, **k: _Gone())["state"] == H.ERROR
    # Un modelo puesto a mano no se cambia solo: se explica qué hacer.
    manual = _providers(OPENROUTER_API_KEY="sk-or-x", OPENROUTER_MODEL="openai/gpt-oss-120b:free")[2]
    r = H.chat(manual, [], post=lambda *a, **k: _Gone())
    assert not r["ok"] and "retiró la versión gratis" in r["detail"] and "no lo pongas" in r["detail"]
    assert "Cambiar el modelo" in r["detail"] and "openrouter/free" in r["detail"]
    # La privacidad cerrada no es «modelo retirado»: no se cambia de modelo, se dice cómo abrirla.
    assert not H._model_gone(json.dumps({"error": {"message": "No endpoints found matching your data policy"}}))


def test_no_free_deepseek_today_is_said_without_calling():
    H._FREE.update(t=time.time(), ids=["google/gemma-4-31b-it:free"])
    ds = _providers(OPENROUTER_API_KEY="sk-or-x")[3]
    assert ds["model"] == "(ninguno gratis ahora)" and "ningún modelo DeepSeek gratis" in ds["unavailable"]
    s = H.ping(ds, post=lambda *a, **k: pytest.fail("no debe llamar"))
    assert s["state"] == H.ERROR and "de pago necesita saldo" in s["detail"]


def test_free_list_is_read_once_and_retried_after_a_failure(openrouter_free_list, monkeypatch):
    for _ in range(3):
        _providers(OPENROUTER_API_KEY="sk-or-x")
    assert len(openrouter_free_list) == 1                                        # una lectura cada 6 h
    H._FREE.update(t=0.0, ids=None)
    monkeypatch.setattr(H, "_models_json", lambda get, timeout=8: (_ for _ in ()).throw(OSError("sin red")))
    ds = _providers(OPENROUTER_API_KEY="sk-or-x")[3]
    assert ds["model"] == "deepseek/deepseek-v4-flash:free" and not ds.get("auto_model")   # el de reserva
    assert H.free_models(now=H._FREE["t"] + 60) is None                          # no insiste en cada recarga
    monkeypatch.setattr(H, "_models_json", lambda get, timeout=8: {"data": [{"id": "deepseek/n:free", "created": 1,
                                                                             "pricing": {"prompt": "0", "completion": "0"}}]})
    assert H.free_models(now=H._FREE["t"] + H.FREE_LIST_RETRY + 1) == ["deepseek/n:free"]


def test_router_says_which_free_model_answered_and_same_model_votes_once():
    def post(url, headers, json, timeout):
        served = "deepseek/deepseek-v4-flash-0731:free" if json["model"] == "openrouter/free" else json["model"]
        return _Resp(body={"model": served, "choices": [{"message": {"content": "Hay que analizar.\nACCIÓN: analizar"}}]})

    ps = [p for p in _providers(OPENROUTER_API_KEY="sk-or-x") if p["api_key"]]
    out = {r["id"]: r for r in H.ask_all(ps, "x", DATA, post=post)}
    assert out["openrouter"]["model"] == "openrouter/free"
    assert out["openrouter"]["served_model"] == out["deepseek"]["served_model"] == "deepseek/deepseek-v4-flash-0731:free"
    assert H.consensus(list(out.values())) == ("auto_analyze", 1, 2)             # mismo modelo: un voto


def test_verify_checks_cited_data_and_references_against_the_context():
    ctx = json.dumps({"ANALISIS": {"medidas": {"beta_espectro": 2.61}, "semaforo": {"nivel": "robust"}},
                      "HIPOTESIS": {"tabla": [{"referencia": "Arzoumanian et al. 2011, A&A 529, L6"}]}})
    assert H.verify("β = 2,61 [beta_espectro]; 🟢 [semaforo]; ver Arzoumanian et al. 2011.", ctx) == []
    bad = H.verify("β = 2,61 [medidas.beta_inventada], como mostró Pérez et al. 2011.", ctx)
    assert any("[medidas.beta_inventada]" in i for i in bad) and any("Pérez" in i for i in bad)
    assert H.verify("Más en la [web](https://example.org).", ctx) == []          # enlaces: no son citas


def test_rules_and_format_are_explicit():
    for needle in ("No lo sé con estos datos", "entre corchetes", "referencias que aparecen en la tabla",
                   "**Qué muestra:**", "**Qué NO se puede afirmar:**", "**Siguiente prueba:**", "ACCIÓN:"):
        assert needle in H.SYSTEM_PROMPT, needle
    assert len(H.QUICK_QUESTIONS) >= 4


@pytest.mark.parametrize("key,name", [("nvapi-abc", "NVIDIA_API_KEY"), ("  sk-or-v1-x", "OPENROUTER_API_KEY"),
                                      ("gsk_123", "GROQ_API_KEY"), ("AIzaSy", "GEMINI_API_KEY"), ("hola", None)])
def test_key_prefix_detection(key, name):
    assert H.detect_key_name(key) == name
    assert H.providers_for_key("OPENROUTER_API_KEY") == ["OpenRouter", "DeepSeek"]


def test_answers_arrive_as_they_finish():
    import time as _t

    def post(url, headers, json, timeout):
        _t.sleep(0.3 if "nvidia" in url else 0.0)                 # NVIDIA tarda más
        return _Resp(content="ok\nACCIÓN: ninguna")

    ps = _providers(NVIDIA_API_KEY="a", GROQ_API_KEY="b")
    order = [r["id"] for r in H.ask_iter(ps, "x", DATA, post=post)]
    assert order == ["groq", "nvidia"]                            # llega primero la rápida
    assert [r["id"] for r in H.ask_all(ps, "x", DATA, post=post)] == ["nvidia", "groq"]   # orden fijo al final


def test_context_carries_more_interpretation():
    card = {"source": {"filename": "f.fits", "center_ra_deg": 315.4, "center_dec_deg": 68.2},
            "analysis": {"crop": {"x0": 100, "y0": 50, "side": 400}, "psf": {"lambda_um": 21.0}},
            "analytics": {"spectrum": {"fit": {"beta": 2.6, "beta_raw": 3.7, "psf_corrected": True}},
                          "scales": {"flatness": [36.5], "flatness_null_q95": [12.4]},
                          "local_map": {"z": [[1, 251.1], [3, -1]], "tile_px": [200, 200]}},
            "descriptors": {"anisotropy": {"structure_pa_deg": 47.0, "orientation_coherence": 0.4,
                                           "orientation_p": 0.01}},
            "discovery_controls": {"point_sources_masked": {"n_masked": 42, "masked_fraction": 0.12, "valid": False}},
            "discovery_gate": {"level": "unconfirmed", "checks": []}}
    a = json.loads(H.build_context(card))["ANALISIS"]
    assert a["difraccion"]["beta_sin_corregir"] == 3.7 and a["difraccion"]["lambda_um"] == 21.0
    assert a["direccion"]["estructuras_en_el_cielo_PA"] == 47.0 and a["direccion"]["p_direccion"] == 0.01
    assert a["control_estrellas"]["valid"] is False and a["intermitencia"]["curtosis_1px"] == 36.5
    assert a["zonas_mas_fuertes"][0] == {"x": [300, 500], "y": [50, 250], "z": 251.1}
    assert a["centro_RA_Dec"] == [315.4, 68.2]


@pytest.mark.parametrize("code,message,needle", [
    (404, "No endpoints found matching your data policy (Free model publication). Configure: "
          "https://openrouter.ai/settings/privacy", "openrouter.ai/settings/privacy"),
    (401, "No auth credentials found", "vuelve a copiarla entera"),
    (404, "deepseek/deepseek-r1-0528:free is not a valid model ID", "ese modelo ya no está disponible"),
    (429, "Rate limit exceeded: free-models-per-day", "límite gratuito"),
])
def test_error_text_says_what_to_do_and_quotes_the_service(code, message, needle):
    """Caso real: clave de OpenRouter pegada en la app y 🔴 en OpenRouter y DeepSeek sin ver el motivo."""
    text = H._error_text(code, json.dumps({"error": {"message": message, "code": code}}))
    assert needle in text and "(%d)" % code in text and "mensaje del servicio" in text


def test_error_text_never_repeats_keys_or_raw_json():
    assert "sk-or-v1" not in H._error_text(400, "falló con sk-or-v1-abcdef123456")
    assert H._error_text(401, json.dumps({"choices": []})) == "clave no válida o sin permiso (401)"
