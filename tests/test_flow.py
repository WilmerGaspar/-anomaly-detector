"""🧭 Revisión del flujo (flow.py) y órdenes de trabajo: reglas fijas, órdenes de una lista cerrada que
proponen las reglas y las IA, la persona las elige y se ejecutan en cola. De punta a punta en la app real
con MAST e IA simuladas."""
import json
from pathlib import Path

import requests

import ai_hub as H
import flow
from test_autopilot_app import fake_mast  # noqa: F401  (fixture: MAST simulado)

ROOT = Path(__file__).resolve().parents[1]
MAST = flow.MAST_KIND


def _card(p=0.01, n=99, filt="F2100W", target="NGC 7023", psf=True, limited=False, checks=()):
    return {"source": {"filename": "jw_x_i2d.fits", "filter": filt, "target": target},
            "analysis": {"null": {"n_simulations": n}, "psf": {"lambda_um": 21.0}},
            "structure_test": {"p_value": p},
            "analytics": {"spectrum": {"fit": {"psf_corrected": psf, "psf_limited": limited}}}}, \
        {"icon": "🟠", "title": "Sin confirmar", "checks": list(checks)}


def _state(**kw):
    s = {"field_loaded": None, "card": None, "gate": None, "log": [], "n_null": 99, "source_kind": MAST,
         "has_search": False, "obj": "NGC 7023"}
    s.update(kw)
    return s


def _by(items):
    return {it["paso"]: it for it in items}


def test_without_file_the_first_order_is_load_and_analyse():
    items = flow.review(_state())
    assert _by(items)["Archivo cargado"]["estado"] == flow.TODO and _by(items)["Archivo cargado"]["orden"] == "auto"
    assert flow.suggested(items)[0] == "auto" and "known_control" in flow.suggested(items)
    # Archivo propio: no hay búsqueda en MAST que hacer.
    assert "auto" not in flow.feasible(_state(source_kind="Archivo propio"))
    loaded = flow.review(_state(field_loaded="f_i2d.fits"))
    assert _by(loaded)["Análisis de la región"]["orden"] == "auto_analyze"


def test_a_result_is_reviewed_like_a_demanding_reviewer():
    card, gate = _card(checks=[{"check": "Dato científico (imagen calibrada)", "ok": False, "detail": "es un PNG"},
                               {"check": "Región sin zonas vacías", "ok": False, "detail": "3 % vacío."},
                               {"check": "Señal frente al nulo (FDR o crestas)", "ok": True, "detail": "2 de 4"}])
    log = [{"target": "NGC 7023", "filter": "F2100W"}]
    st = _state(field_loaded="jw_x_i2d.fits", card=card, gate=gate, log=log, has_search=True)
    by = _by(flow.review(st))
    assert by["Dato científico (imagen calibrada)"]["orden"] == "other_filter"
    assert "2. Región" in by["Región sin zonas vacías"]["detalle"] and by["Región sin zonas vacías"]["orden"] is None
    assert by["Señal frente al nulo (FDR o crestas)"]["estado"] == flow.OK
    # p en el mínimo posible con 99 subrogados: pedir más.
    assert by["Precisión del nulo"]["orden"] == "more_null" and "199" in by["Precisión del nulo"]["detalle"]
    # Un solo filtro de NGC 7023: confirmar con otro.
    assert by["Otro filtro del mismo objeto"]["orden"] == "other_filter"
    assert by["Referencias para Novedad"]["estado"] == flow.INFO and "1 de 5" in by["Referencias para Novedad"]["detalle"]
    assert by["Control conocido"]["orden"] == "known_control"
    assert by["Difracción del telescopio"]["estado"] == flow.OK
    assert flow.suggested(flow.review(st))[:3] == ["other_filter", "more_null", "known_control"]


def test_what_is_already_done_is_not_asked_again():
    card, gate = _card(p=0.2)
    log = [{"target": "NGC 7023", "filter": "F1000W"}, {"target": "Crab Nebula", "filter": "F502N"}] + \
          [{"target": "x%d" % i, "filter": "F1"} for i in range(3)]
    by = _by(flow.review(_state(field_loaded="f", card=card, gate=gate, log=log, has_search=True)))
    assert by["Precisión del nulo"]["estado"] == flow.OK
    assert by["Otro filtro del mismo objeto"]["estado"] == flow.OK and "F1000W, F2100W" in by["Otro filtro del mismo objeto"]["detalle"]
    assert by["Control conocido"]["estado"] == flow.OK and by["Referencias para Novedad"]["estado"] == flow.OK
    card2, gate2 = _card(limited=True, psf=False)
    assert _by(flow.review(_state(card=card2, gate=gate2)))["Difracción del telescopio"]["estado"] == flow.FAIL


def test_null_levels_and_pipeline_order():
    assert [flow.next_null(n) for n in (49, 99, 199, 499)] == [99, 199, 499, None]
    assert flow.in_pipeline_order(["known_control", "other_filter", "auto_analyze"]) == \
        ["auto_analyze", "other_filter", "known_control"]
    assert "more_null" not in flow.feasible(_state(field_loaded="f", n_null=499))


def test_ai_orders_come_only_from_the_closed_list():
    text, orders = H.split_orders("**Qué muestra:** algo.\nORDEN: otro_archivo — confirmar con otro filtro [filtro]\n"
                                  "**ORDEN:** mas_subrogados: p en el mínimo [prueba_frente_al_nulo]\n"
                                  "ORDEN: borrar_todo — no existe\nORDEN: otro_archivo — repetida\nACCIÓN: otro_archivo")
    assert orders == [("other_filter", "confirmar con otro filtro [filtro]"), ("more_null", "p en el mínimo [prueba_frente_al_nulo]")]
    assert "ORDEN" not in text and "ACCIÓN" in text


def test_order_votes_count_only_verified_answers_once_per_model():
    def r(name, orders, issues=(), served=None):
        return {"id": name, "name": name, "ok": True, "orders": orders, "issues": list(issues), "action": None,
                "served_model": served}
    out = H.consensus_orders([r("NVIDIA", [("other_filter", "a")]), r("Groq", [("other_filter", "b"), ("more_null", "c")]),
                              r("OpenRouter", [("other_filter", "d")], served="x/m:free"),
                              r("DeepSeek", [("other_filter", "e")], served="x/m"),          # mismo modelo
                              r("Gemini", [("known_control", "f")], issues=["cita números"])])
    assert out["other_filter"]["votes"] == 3 and out["more_null"]["votes"] == 1
    assert out["known_control"]["votes"] == 0 and out["known_control"]["reasons"] == [("Gemini", "f", False)]


def test_context_carries_the_flow_review_with_ai_keys():
    items = flow.review(_state())
    ctx = json.loads(H.build_context(None, actions=("auto", "known_control"), flow=items))
    assert ctx["ACCIONES_DISPONIBLES"] == ["ninguna", "hazlo_por_mi", "control_conocido"]
    assert ctx["REVISION_FLUJO"][0] == {"paso": "Archivo cargado", "estado": "pendiente",
                                        "detalle": items[0]["detalle"], "orden": "hazlo_por_mi"}


class _Resp:
    def __init__(self, content):
        self.status_code = 200
        self._body = {"choices": [{"message": {"content": content}}]}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


def test_welcome_window_orders_and_queue_end_to_end(monkeypatch, fake_mast):  # noqa: F811
    """Al entrar con una IA 🟢 y sin imagen: ventana «¿Listo para trabajar?» → «Sí» EMPIEZA A TRABAJAR (busca,
    carga y analiza) → el resultado se ve en «Órdenes de trabajo» → las IA revisan el flujo con el resultado y
    proponen órdenes (lista cerrada, verificadas) → la persona marca una y la ejecuta → nueva revisión.
    Caso real (captura del 10-oct-2026): «Sí» solo revisaba y, sin imagen, no salía ningún resultado."""
    for k in ("OPENROUTER_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-SECRETO-7")
    asked = []

    def post(url, headers=None, json=None, timeout=None, **kw):
        if json["max_tokens"] <= 16:
            return _Resp("OK")
        content = json["messages"][-1]["content"]
        asked.append(content.split("PREGUNTA: ", 1)[1])
        assert "REVISION_FLUJO" in content and '"ANALISIS"' in content          # revisa con el resultado
        return _Resp("**Qué muestra:** p en el mínimo posible [prueba_frente_al_nulo].\n**Siguiente prueba:** más "
                     "subrogados.\nACCIÓN: mas_subrogados\nORDEN: mas_subrogados — afinar p [prueba_frente_al_nulo]")

    monkeypatch.setattr(requests, "post", post)
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    yes = next(b for b in at.button if b.key == "ai_welcome_yes")
    assert "busca, carga y analiza JWST · NGC 7023" in yes.label and not asked      # saludar no gasta consultas
    yes.click().run()

    def until(cond, n=8):
        for _ in range(n):
            if cond():
                return
            at.run()

    until(lambda: asked)
    assert not at.exception, [e.value for e in at.exception]
    assert fake_mast["download"] == ["mast:ok"] and len(at.session_state["guide_log"]) == 1
    assert at.session_state["order_log"][0].startswith("✔ Hazlo por mí")
    assert asked == [H.REVIEW_QUESTION]                                          # revisión con el resultado
    assert any("📊 **Resultado actual:** jw_prueba_i2d.fits · F200W" in str(m.value) for m in at.markdown)
    picks = {c.key.rsplit("_", 1)[0]: c for c in at.checkbox if (c.key or "").startswith("order_pick_")}
    pick = picks["order_pick_more_null"]
    assert pick.value is True and "🤖 1 IA" in pick.label
    # Lo que las IA no votaron (aquí el control conocido, que solo propone la regla) sale sin marcar.
    assert picks["order_pick_known_control"].value is False and "🧠 regla" in picks["order_pick_known_control"].label
    for k, c in picks.items():                                                   # solo esa orden
        if k != "order_pick_more_null":
            c.uncheck()
    next(b for b in at.button if b.key == "orders_run").click().run()
    until(lambda: len(asked) >= 2)
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["n_null"] == 199 and at.session_state["analysis_count"] == 2
    assert at.session_state["order_log"] == ["✔ Repetir con más subrogados (p más preciso)"]
    assert len(asked) == 2                                                       # y otra revisión al terminar
    assert "SECRETO" not in " ".join(str(m.value) for m in at.markdown)
