"""🧠 Guía: asistente por reglas (sin IA de pago, sin conexión, gratis).

Mira en qué punto está la app y dice el siguiente paso, con botones para hacerlo de un clic.
Tras un análisis traduce el semáforo a lenguaje claro y propone el siguiente procedimiento.
Incluye un glosario que solo responde lo que está escrito aquí: no improvisa, así que no puede
inventar. Todo es lógica pura (sin Streamlit) para poder probarla.
"""
from __future__ import annotations

import unicodedata

MAX_OBS_TRY = 6          # observaciones que el piloto automático revisa antes de rendirse


# ---------------------------------------------------------------- elegir archivo

def pick_product(prods):
    """Mejor archivo para analizar: imagen 2D, que quepa en la nube. `list_fits_products` ya
    ordena por tipo (i2d/drc/drz primero) y tamaño, así que vale el primero que cumpla."""
    for p in prods or []:
        # is_public False: acceso exclusivo, MAST da 401 (solo el equipo del programa).
        if p.get("is_image") and not p.get("too_big") and p.get("uri") and p.get("is_public", True):
            return p
    return None


def rows_to_try(rows, done_filters=(), done_obs=()):
    """Observaciones en orden de prueba: primero las de filtros aún no analizados (para
    confirmar en otro filtro), sin repetir observaciones ya usadas."""
    done_filters = {str(f).upper() for f in done_filters if f}
    # Sin las de acceso exclusivo (🔒): MAST las lista pero la descarga da 401.
    fresh = [r for r in rows or [] if r.get("obs_id") not in set(done_obs) and r.get("is_public", True)]
    # MAST puede dar "F444W;F405N" y la cabecera "F405N": vale con que aparezca.
    new_filter = [r for r in fresh if not any(f in str(r.get("filters") or "").upper() for f in done_filters)]
    rest = [r for r in fresh if r not in new_filter]
    return (new_filter + rest)[:MAX_OBS_TRY]


# ---------------------------------------------------------------- interpretar el resultado

def _failed(card):
    return [c for c in (card.get("discovery_gate") or {}).get("checks", []) if not c.get("ok")]


def explain_result(card, level=None, mast=True):
    """(título, explicación, procedimientos) en lenguaje claro para un análisis. Con mast=False
    (archivo propio o URL) no se mencionan los botones que solo existen con MAST."""
    other = ("Pulsa ▶ Otro archivo del mismo objeto (otra exposición u otro detector)." if mast else
             "Busca otra exposición del mismo objeto en MAST (Origen: Archivo MAST).")
    other_filter = ("Pulsa ▶ Otro archivo del mismo objeto (prueba primero otro filtro): un resultado real se repite."
                    if mast else
                    "Analiza el mismo objeto en otro filtro (búscalo en MAST): un resultado real se repite.")
    gate = card.get("discovery_gate") or {}
    level = level or gate.get("level")
    failed = _failed(card)
    names = {c["check"] for c in failed}
    state = card.get("state")
    if level == "invalid":
        if "Región sin zonas vacías" in names:
            return ("🔴 Región no válida", "La región incluye bordes vacíos del mosaico, que el test confunde con estructura.",
                    ["Mueve el recuadro hacia el centro de la imagen (los valores por defecto ya evitan los bordes).",
                     "Vuelve a pulsar Analizar."])
        if "Sin artefacto de instrumento" in names or state == "instrument_artifact":
            return ("🔴 Huella del detector", "El detector deja un patrón alineado con sus ejes que se puede confundir con "
                    "estructura. No es cielo.",
                    [other, "Si se repite en todos, es propio del instrumento: cambia de objeto."])
        return ("🔴 Dato no científico", "El archivo no es una imagen calibrada (por ejemplo, un PNG o un dato crudo).",
                ["Usa un FITS calibrado: i2d (JWST) o drz/drc (Hubble). El piloto automático ya los elige."])
    if level == "none":
        text = "No hay organización que el ruido equivalente no explique."
        if gate.get("insensitive"):
            text += (" Ojo: la región es menos irregular que su nulo (muchas estrellas); aquí la prueba pierde "
                     "sensibilidad y un ⚪ no descarta estructura.")
        return ("⚪ Sin estructura", text,
                ["Prueba otra región de la misma imagen, con menos estrellas.",
                 other.replace("Pulsa", "O pulsa").replace("Busca", "O busca"), "O prueba otro objeto."])
    if level == "explained":
        return ("🟡 Lo explican fuentes conocidas", "Hay señal, pero desaparece al tapar las estrellas: la causan ellas, "
                "no una estructura extendida.",
                ["Mueve el recuadro a una zona con nubes o filamentos y pocas estrellas.",
                 "Si toda la imagen está llena de estrellas, prueba otro objeto (una nebulosa)."])
    if level == "unconfirmed":
        return ("🟠 Sin confirmar", "Hay señal, pero faltan controles: hay tantas fuentes que no se pueden tapar sin "
                "estropear la imagen.",
                ["Reduce o mueve la región a una zona con menos estrellas (tapar debe quedar por debajo del 10 %).",
                 "Vuelve a analizar."])
    if level in ("robust", "pioneer"):
        hyp = card.get("hypotheses") or {}
        title = "🟣 Atípico y robusto" if level == "pioneer" else "🟢 Estructura robusta"
        text = ("La región tiene organización real que no explican ni el ruido ni las estrellas. No es un "
                "descubrimiento todavía: hay que repetirlo con datos independientes.")
        if hyp.get("active"):
            text += " " + hyp.get("title", "") + "."
        return (title, text,
                ["Abre la pestaña 🧪 Hipótesis: la columna «pregunta» dice qué prueba decide cada una.",
                 other_filter,
                 "Descarga el JSON (está en los resultados) para guardarlo.",
                 "Con 5 análisis en la bitácora, la pestaña Novedad dirá si este es distinto de los demás."])
    return ("Resultado", "Revisa el semáforo en los resultados.", [])


# ---------------------------------------------------------------- siguiente paso

def next_step(st_state):
    """Mensaje de la guía según el estado: dict con 'title', 'text', 'steps' y 'actions'
    (lista de claves de botón: 'auto', 'auto_analyze', 'other_filter')."""
    s = st_state
    if s.get("mode") == "radio":
        return {"title": "Modo radio", "text": "Busca firmas de púlsar, ráfaga o portadora en datos de radio.",
                "steps": ["Genera un ejemplo de la biblioteca para ver cómo responde cada detector.",
                          "O carga tu archivo (.npy, .csv o .fits) y pulsa Buscar firmas."], "actions": []}
    card = s.get("card")
    if card:
        mast = s.get("source_kind") == "Archivo MAST (STScI)" and bool(s.get("has_search"))
        title, text, steps = explain_result(card, s.get("level"), mast=mast)
        actions = ["other_filter"] if mast else []
        return {"title": title, "text": text, "steps": steps, "actions": actions}
    if s.get("field_loaded"):
        return {"title": "Paso 3 · Analizar", "text": "Imagen cargada: %s." % s["field_loaded"],
                "steps": ["La región por defecto ya evita los bordes vacíos.",
                          "Pulsa Analizar (tarda entre 20 s y 1 min)."], "actions": ["auto_analyze"]}
    kind = s.get("source_kind")
    if kind == "Archivo MAST (STScI)":
        if s.get("has_prods"):
            return {"title": "Paso 2 · Elegir archivo", "text": "Elige un archivo i2d (JWST) o drz/drc (Hubble) de menos de "
                    "400 MB y pulsa Cargar.", "steps": [], "actions": ["auto"]}
        if s.get("has_search"):
            return {"title": "Paso 2 · Elegir observación", "text": "Haz clic en una fila y pulsa «Listar archivos FITS».",
                    "steps": ["O deja que el piloto automático elija la primera observación con una imagen válida."],
                    "actions": ["auto"]}
        obj = s.get("obj") or "un objeto"
        return {"title": "Paso 1 · Buscar", "text": "Misión %s, objeto %s." % (s.get("mission") or "—", obj),
                "steps": ["▶ Hazlo por mí busca en MAST, elige el mejor archivo, lo carga y lo analiza en un clic.",
                          "Para empezar, un control conocido: HST · Crab Nebula (filamentos) o HST · M16 (pilares)."],
                "actions": ["auto"]}
    if kind == "Archivo propio":
        return {"title": "Paso 1 · Subir archivo", "text": "Arrastra un FITS de hasta 200 MB.",
                "steps": ["Para archivos más grandes usa MAST o URL directa."], "actions": []}
    return {"title": "Paso 1 · URL", "text": "Pega la URL de un FITS y pulsa Descargar.", "steps": [], "actions": []}


# ---------------------------------------------------------------- bitácora

def log_entry(card, key):
    src = card.get("source") or {}
    gate = card.get("discovery_gate") or {}
    return {"key": list(key), "file": src.get("filename"), "filter": src.get("filter"), "target": src.get("target"),
            "level": gate.get("level"), "icon": gate.get("icon"), "card": card}


def add_to_log(log, entry):
    """Añade o reemplaza (mismo archivo, extensión y región): repetir un análisis no duplica."""
    log = [e for e in (log or []) if e["key"] != entry["key"]]
    return log + [entry]


def references_from_log(log, current_key):
    """Tarjetas de la bitácora como referencias para Novedad, sin el análisis actual."""
    return [e["card"] for e in log or [] if e["key"] != list(current_key)]


def log_summary(log):
    counts = {}
    for e in log or []:
        counts[e.get("icon") or "·"] = counts.get(e.get("icon") or "·", 0) + 1
    return " ".join("%s %d" % (k, v) for k, v in counts.items())


# ---------------------------------------------------------------- glosario

GLOSSARY = [
    (("semaforo", "colores", "color"),
     "🔴 no válido · ⚪ sin estructura · 🟡 lo explican fuentes conocidas · 🟠 faltan controles · 🟢 estructura robusta "
     "· 🟣 robusta y distinta de tus análisis anteriores. Ningún color es un descubrimiento por sí solo."),
    (("nulo", "subrogado", "iaaft"),
     "El nulo son imágenes falsas con el mismo brillo y el mismo espectro que la real, pero con la estructura "
     "barajada (IAAFT). Si la real se distingue de todas ellas, hay organización que el ruido no explica."),
    (("fdr", "benjamini", "falsos positivos"),
     "FDR (Benjamini-Hochberg) corrige por hacer varias pruebas a la vez, para que no salgan positivos por azar. "
     "«3/4 pasan» = 3 de las 4 pruebas superan esa corrección."),
    (("beta", "β", "pendiente", "espectro"),
     "β es la pendiente del espectro de potencia: cómo se reparte el brillo entre escalas grandes y pequeñas. "
     "≈ 3.7 se asocia a turbulencia de Kolmogorov, 2–3 a turbulencia con choques. El desenfoque del telescopio y "
     "el ruido también la cambian."),
    (("lacunaridad",),
     "Lacunaridad: cuánto hueco hay entre las estructuras. ≈ 1 = repartido sin huecos; mayor = grumos separados "
     "por vacíos."),
    (("multifractal", "multifractalidad"),
     "Multifractalidad: si el brillo se concentra en pocas zonas (alta) o se reparte por igual (≈ 0)."),
    (("betti", "huecos", "componentes", "topologia"),
     "Números de Betti: β0 = piezas separadas; β1 = huecos rodeados de emisión (posibles cáscaras o burbujas)."),
    (("psf", "estrella", "estrellas", "difraccion", "picos", "enmascarar", "mascara"),
     "Las estrellas dejan su forma (PSF) y picos de difracción que el test puede tomar por estructura. La app los "
     "tapa y repite la prueba: si la señal desaparece, la causaban ellas (🟡)."),
    (("i2d", "drz", "drc", "cal", "archivo", "fits"),
     "Elige imágenes calibradas: i2d (JWST) o drz/drc (Hubble). cal y flt son exposiciones sueltas; uncal y raw "
     "son datos crudos que no sirven para concluir."),
    (("mast",),
     "MAST es el archivo de datos de Hubble y JWST (STScI). La app lo consulta por su API, gratis."),
    (("novedad", "pionero", "morado"),
     "Novedad compara este análisis con tus análisis anteriores (los de la bitácora o JSON que subas, mínimo 5). "
     "«Distinto de lo que ya analizaste», no «nuevo para la ciencia»."),
    (("hipotesis", "mecanismo"),
     "La pestaña 🧪 Hipótesis (solo con 🟢 o 🟣) compara las medidas con mecanismos físicos publicados y dice qué "
     "prueba confirmaría o descartaría cada uno."),
    (("z", "sigma", "p valor", "valor p"),
     "z: cuántas desviaciones se aleja la región de su nulo. p: probabilidad de verlo por azar; con 99 subrogados, "
     "el mínimo posible es 0.01."),
    (("region", "recuadro", "tamano"),
     "La región es el recuadro que se analiza (máximo 1024 px). Para cubrir más área, analiza varias regiones."),
]


def _norm(text):
    t = unicodedata.normalize("NFD", str(text).lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def answer(question):
    """Respuesta del glosario o None. Solo devuelve textos de GLOSSARY: no improvisa."""
    q = " %s " % _norm(question).replace("?", " ").replace("¿", " ").replace(",", " ")
    best, score = None, 0
    for keys, text in GLOSSARY:
        s = sum(1 for k in keys if (" %s " % k) in q or (len(k) > 3 and k in q))
        if s > score:
            best, score = text, s
    return best


GLOSSARY_TOPICS = "semáforo, nulo, FDR, β, lacunaridad, Betti, estrellas, archivos i2d/drz, MAST, Novedad, Hipótesis, z, región"
