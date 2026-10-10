"""🧭 Revisión del flujo y órdenes de trabajo.

Lo que un revisor exigente comprobaría en cada paso, hecho con reglas fijas de la app (no con IA, así
que no puede inventar nada): archivo, análisis, cada comprobación del semáforo, precisión del nulo,
otro filtro del mismo objeto, referencias para Novedad, un control conocido y la difracción. Cada punto
pendiente propone una ORDEN de una lista cerrada; las IA pueden proponer de esa misma lista y la
persona elige cuáles se ejecutan.
"""
from __future__ import annotations

MAST_KIND = "Archivo MAST (STScI)"
NULL_LEVELS = (49, 99, 199, 499)
CONTROL = ("HST", "Crab Nebula")          # control positivo (filamentos), de la lista de objetos de la app
_CONTROL_NAMES = ("crab", "m1 ", "m 1 ", "m16", "m 16", "eagle", "pillars")
NOVELTY_MIN = 5
# Orden natural de ejecución: cargar, analizar, afinar el nulo, confirmar con otro filtro y, al final,
# el control (cambia de objeto).
PIPELINE = ("auto", "auto_analyze", "more_null", "other_filter", "known_control")

OK, FAIL, TODO, INFO = "ok", "falla", "pendiente", "info"
ICON = {OK: "✅", FAIL: "❌", TODO: "⏳", INFO: "ℹ️"}


def next_null(n):
    """Siguiente número de subrogados (más precisión para p), o None si ya es el máximo."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return NULL_LEVELS[1]
    bigger = [x for x in NULL_LEVELS if x > n]
    return bigger[0] if bigger else None


def _norm_target(t):
    return " ".join(str(t or "").lower().replace("-", " ").replace("_", " ").split())


def _is_control(target):
    t = " " + _norm_target(target) + " "
    return any(c in t for c in _CONTROL_NAMES)


def feasible(state):
    """Órdenes que se pueden ejecutar ahora (claves de la guía)."""
    mast = state.get("source_kind") == MAST_KIND
    out = []
    if mast and state.get("obj") and not state.get("field_loaded"):
        out.append("auto")
    if state.get("field_loaded"):
        out.append("auto_analyze")
        if next_null(state.get("n_null")):
            out.append("more_null")
    if mast and state.get("has_search") and state.get("card"):
        out.append("other_filter")
    out.append("known_control")
    return out


def review(state):
    """Lista de puntos {paso, estado, detalle, orden}. `state`: field_loaded, card, gate, log, n_null,
    source_kind, has_search, obj."""
    can = feasible(state)
    items = []

    def add(paso, estado, detalle, orden=None):
        items.append({"paso": paso, "estado": estado, "detalle": detalle, "orden": orden if orden in can else None})

    card, gate, log = state.get("card"), state.get("gate") or {}, state.get("log") or []
    field = state.get("field_loaded")
    if not field:
        add("Archivo cargado", TODO, "Todavía no hay imagen: hay que buscar y cargar un FITS calibrado.", "auto")
    else:
        add("Archivo cargado", OK, str(field))
    if field and not card:
        add("Análisis de la región", TODO, "Imagen cargada sin analizar.", "auto_analyze")
    if not card:
        add("Control conocido", INFO if not any(_is_control(e.get("target")) for e in log) else OK,
            "Analizar un control conocido (%s · %s, filamentos) dice si el método responde como se espera."
            % CONTROL, "known_control")
        return items

    src, an = card.get("source") or {}, card.get("analysis") or {}
    add("Análisis de la región", OK, "Hecho: %s %s." % (gate.get("icon") or "", gate.get("title") or "resultado"))
    for c in gate.get("checks") or []:
        orden = None
        detail = str(c.get("detail") or "")
        if not c.get("ok") and c.get("check", "").startswith("Dato científico"):
            orden = "other_filter"
        if not c.get("ok") and c.get("check", "").startswith("Región sin zonas"):
            detail += " Mueve o reduce la región en «2. Región»."
        add(c.get("check") or "Comprobación", OK if c.get("ok") else FAIL, detail, orden)

    st_ = card.get("structure_test") or {}
    p = st_.get("p_value")
    n = (an.get("null") or {}).get("n_simulations") or state.get("n_null")
    try:
        p_min = 1.0 / (int(n) + 1)
    except (TypeError, ValueError):
        p_min = None
    if p is not None and p_min and p <= p_min * 1.0001 and next_null(n):
        add("Precisión del nulo", TODO, "p = %.4f es el mínimo posible con %d subrogados: con %d se sabe si es aún menor."
            % (p, int(n), next_null(n)), "more_null")
    elif p is not None and p_min:
        add("Precisión del nulo", OK, "p = %.4f con %d subrogados (mínimo posible %.4f)." % (p, int(n), p_min))

    target = src.get("target") or src.get("filename")
    same = [e for e in log if _norm_target(e.get("target")) == _norm_target(target)] if target else []
    filters = sorted({str(e.get("filter")) for e in same if e.get("filter")} | ({str(src["filter"])} if src.get("filter") else set()))
    if len(filters) < 2:
        add("Otro filtro del mismo objeto", TODO, "Solo %s analizado para %s: una estructura real debe aparecer (o "
            "explicarse por qué no) en otra longitud de onda." % (filters[0] if filters else "un filtro", target or "este objeto"),
            "other_filter")
    else:
        add("Otro filtro del mismo objeto", OK, "Filtros analizados: %s." % ", ".join(filters))

    if len(log) < NOVELTY_MIN:
        add("Referencias para Novedad", INFO, "Novedad compara con tus análisis de la sesión: tienes %d de %d."
            % (len(log), NOVELTY_MIN), "other_filter")
    else:
        add("Referencias para Novedad", OK, "%d análisis en la bitácora." % len(log))

    if any(_is_control(e.get("target")) for e in log) or _is_control(target):
        add("Control conocido", OK, "Hay un control conocido en la bitácora.")
    else:
        add("Control conocido", INFO, "Analizar un control conocido (%s · %s, filamentos) dice si el método responde "
            "como se espera." % CONTROL, "known_control")

    fit = ((card.get("analytics") or {}).get("spectrum") or {}).get("fit") or {}
    if fit.get("psf_limited"):
        add("Difracción del telescopio", FAIL, "β no medible: la difracción no deja escalas suficientes en este filtro.")
    elif fit.get("psf_corrected"):
        add("Difracción del telescopio", OK, "β corregido por la difracción (λ = %s µm)."
            % ((an.get("psf") or {}).get("lambda_um") or "—"))
    else:
        add("Difracción del telescopio", INFO, "β sin corregir: falta la escala de píxel o el filtro en la cabecera.")
    return items


def suggested(items):
    """Órdenes de los puntos pendientes, sin repetir, en el orden de la revisión."""
    out = []
    for it in items:
        if it["estado"] != OK and it.get("orden") and it["orden"] not in out:
            out.append(it["orden"])
    return out


def in_pipeline_order(actions):
    """Las órdenes elegidas, en el orden en que tiene sentido ejecutarlas."""
    return [a for a in PIPELINE if a in set(actions)]


def summary(items):
    done = sum(1 for it in items if it["estado"] == OK)
    return "%d de %d en orden" % (done, len(items))
