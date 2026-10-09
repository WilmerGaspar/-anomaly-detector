"""🧠 Guía por reglas (guide.py): elegir archivo, siguiente paso, interpretación, bitácora y glosario."""
import pytest

import guide as G


def _prod(name, image=True, big=False, uri="mast:x"):
    return {"filename": name, "is_image": image, "too_big": big, "uri": uri, "size_mb": 50}


def test_pick_product_skips_non_images_and_too_big():
    prods = [_prod("a_uncal.fits", image=False), _prod("b_i2d.fits", big=True), _prod("c_i2d.fits"), _prod("d_cal.fits")]
    assert G.pick_product(prods)["filename"] == "c_i2d.fits"
    assert G.pick_product([_prod("x", image=False)]) is None and G.pick_product(None) is None


def test_rows_to_try_prefers_new_filters_and_skips_used():
    rows = [{"obs_id": "o1", "filters": "F200W"}, {"obs_id": "o2", "filters": "F444W;F405N"},
            {"obs_id": "o3", "filters": "F090W"}, {"obs_id": "o4", "filters": "F200W"}]
    # o3 ya usada; F200W y F405N ya analizados -> no queda filtro nuevo, se prueban las demas en orden.
    assert [r["obs_id"] for r in G.rows_to_try(rows, ["F200W", "F405N"], ["o3"])] == ["o1", "o2", "o4"]
    # Solo F200W analizado -> primero los filtros nuevos (o2, o3), luego el resto.
    assert [r["obs_id"] for r in G.rows_to_try(rows, ["F200W"])] == ["o2", "o3", "o1", "o4"]
    assert len(G.rows_to_try([dict(r, obs_id=str(i)) for i, r in enumerate(rows * 5)])) == G.MAX_OBS_TRY


def _card(level, checks=(), insensitive=False):
    return {"discovery_gate": {"level": level, "insensitive": insensitive,
                               "checks": [{"check": c, "ok": False} for c in checks]},
            "source": {"filename": "f.fits", "filter": "F200W"}}


@pytest.mark.parametrize("level,checks,expect", [
    ("invalid", ["Región sin zonas vacías"], "Región no válida"),
    ("invalid", ["Sin artefacto de instrumento"], "Huella del detector"),
    ("invalid", ["Dato científico (imagen calibrada)"], "Dato no científico"),
    ("none", [], "Sin estructura"),
    ("explained", ["No la explican las fuentes puntuales"], "fuentes conocidas"),
    ("unconfirmed", ["No la explican las fuentes puntuales"], "Sin confirmar"),
    ("robust", [], "Estructura robusta"),
    ("pioneer", [], "Atípico"),
])
def test_every_level_has_plain_explanation_and_next_steps(level, checks, expect):
    title, text, steps = G.explain_result(_card(level, checks))
    assert expect in title and text and steps


def test_green_never_claims_discovery_and_points_to_confirmation():
    title, text, steps = G.explain_result(_card("robust"))
    assert "No es un descubrimiento" in text
    assert any("otro filtro" in s for s in steps) and any("Hipótesis" in s for s in steps)


def test_insensitive_white_is_explained():
    _, text, _ = G.explain_result(_card("none", insensitive=True))
    assert "no descarta estructura" in text


def test_next_step_follows_the_flow():
    base = {"mode": "image", "source_kind": "Archivo MAST (STScI)", "mission": "HST", "obj": "M16"}
    assert G.next_step(base)["actions"] == ["auto"] and "Buscar" in G.next_step(base)["title"]
    assert "observación" in G.next_step(dict(base, has_search=True))["title"]
    assert "archivo" in G.next_step(dict(base, has_search=True, has_prods=True))["title"]
    s = G.next_step(dict(base, field_loaded="x_drz.fits"))
    assert s["actions"] == ["auto_analyze"] and "x_drz.fits" in s["text"]
    s = G.next_step(dict(base, field_loaded="x_drz.fits", has_search=True, card=_card("robust"), level="robust"))
    assert s["actions"] == ["other_filter"] and "robusta" in s["title"]
    own = G.next_step(dict(base, source_kind="Archivo propio", field_loaded="x", card=_card("none"), level="none"))
    assert own["actions"] == []                                   # sin MAST no hay "otro filtro"
    assert G.next_step({"mode": "radio"})["title"] == "Modo radio"


def test_log_dedupes_and_references_exclude_current():
    k1, k2 = ("a.fits", 10, 1, 0, 0, 1024), ("b.fits", 20, 1, 0, 0, 1024)
    log = G.add_to_log([], G.log_entry(_card("robust"), k1))
    log = G.add_to_log(log, G.log_entry(_card("none"), k1))        # mismo análisis repetido
    log = G.add_to_log(log, G.log_entry(_card("none"), k2))
    assert len(log) == 2 and log[0]["level"] == "none"
    assert len(G.references_from_log(log, k1)) == 1
    assert G.log_summary([{"icon": "🟢"}, {"icon": "🟢"}, {"icon": "⚪"}]) == "🟢 2 ⚪ 1"


@pytest.mark.parametrize("q,needle", [("¿Qué es β?", "pendiente"), ("que es la beta", "pendiente"),
                                      ("¿qué significa el semáforo?", "🔴"), ("lacunaridad?", "hueco"),
                                      ("qué archivo uso, drz o i2d", "calibradas"), ("que es el nulo IAAFT", "barajada")])
def test_glossary_answers_known_topics(q, needle):
    assert needle in G.answer(q)


def test_glossary_does_not_improvise():
    assert G.answer("¿hay vida en Marte?") is None
    assert G.answer("receta de pan") is None


def test_no_button_mentioned_without_mast():
    for level, checks in (("robust", ()), ("none", ()), ("invalid", ["Sin artefacto de instrumento"])):
        _, _, steps = G.explain_result(_card(level, checks), mast=False)
        assert not any("▶" in s for s in steps), steps
        _, _, steps = G.explain_result(_card(level, checks), mast=True)
        assert any("▶" in s for s in steps), steps


def test_steps_only_name_real_buttons():
    import re
    labels = {"▶ Hazlo por mí", "▶ Analizar por mí", "▶ Otro archivo del mismo objeto"}
    for level in ("invalid", "none", "explained", "unconfirmed", "robust", "pioneer"):
        for checks in ((), ["Sin artefacto de instrumento"], ["Región sin zonas vacías"]):
            _, _, steps = G.explain_result(_card(level, checks), mast=True)
            for s in steps:
                for m in re.findall(r"▶ [^(:.]+", s):
                    assert any(m.strip().startswith(l) for l in labels), m
