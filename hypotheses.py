"""Hipótesis de formación: de la morfología medida a mecanismos físicos publicados.

Solo se generan con el semáforo en 🟢 o 🟣 (estructura que no explican el ruido ni las fuentes
puntuales). Para cada mecanismo de la tabla se dice:
- estado: compatible / no compatible / no medible con una sola imagen;
- por qué (la medida concreta que lo decide);
- qué predice;
- la pregunta (prueba) que lo confirmaría o lo descartaría;
- la referencia publicada de la relación física.

Reglas para no alucinar:
- "Compatible" no es "explicado": varias hipótesis pueden ser compatibles a la vez y siempre se
  listan las alternativas que una sola imagen no puede descartar (proyección, fuentes no resueltas).
- Los umbrales son rangos amplios tomados de las referencias, no ajustes a un caso.
- Si NINGÚN mecanismo de la tabla es compatible, el patrón se marca como "pregunta abierta":
  no explicado por ESTA tabla, que es corta. No significa física nueva sin las pruebas listadas.
"""
from __future__ import annotations

from discovery import RIDGE_MIN   # mismo umbral de crestas que el semáforo

COMPATIBLE, NOT_COMPATIBLE, NOT_MEASURABLE = "compatible", "no compatible", "no medible"
ICON = {COMPATIBLE: "✅", NOT_COMPATIBLE: "❌", NOT_MEASURABLE: "❔"}


def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    try:
        v = float(d)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def measurements(card):
    """Las medidas del JSON que usa la tabla (todas ya calculadas por el análisis)."""
    sc = (card.get("analytics") or {}).get("scales") or {}
    fl, q95 = sc.get("flatness") or [], sc.get("flatness_null_q95") or []
    # Intermitencia: curtosis de incrementos por encima del 95 % del nulo a 1-2 px.
    inter = None
    if len(fl) >= 2 and len(q95) >= 2:
        inter = bool(fl[0] > q95[0] or fl[1] > q95[1])
    fdr = (card.get("structure_test") or {}).get("fdr") or {}
    per = (card.get("descriptors") or {}).get("periodicity") or {}
    return {
        "beta": _get(card, "analytics", "spectrum", "fit", "beta"),
        "beta_r2": _get(card, "analytics", "spectrum", "fit", "r2"),
        "intermittent": inter,
        "filament_excess": _get(card, "descriptors", "ridges", "filament_excess"),
        # Dirección preferente frente a campos isótropos con el mismo espectro (plugins/anisotropy).
        # Antes se usaba que "aniso" pasara el FDR, pero ese estadístico es el coeficiente de
        # variación de la energía del gradiente (bordes concentrados): no tiene dirección.
        "orientation_p": _get(card, "descriptors", "anisotropy", "orientation_p"),
        "orientation_coherence": _get(card, "descriptors", "anisotropy", "orientation_coherence"),
        "orientation_null_q95": _get(card, "descriptors", "anisotropy", "orientation_null_q95"),
        "structure_direction": _get(card, "descriptors", "anisotropy", "structure_direction_degrees"),
        "anisotropy_index": _get(card, "descriptors", "anisotropy", "anisotropy_index"),
        "betti_1": _get(card, "descriptors", "persistent_homology", "betti_1"),
        "lacunarity": _get(card, "descriptors", "fractal_base", "lacunarity"),
        "multifractality": _get(card, "descriptors", "fractal_base", "multifractality_index"),
        "periodic": bool((per.get("n_significant_peaks") or 0) >= 1 and per.get("lattice_consistent")
                         and not per.get("likely_instrument_artifact")),
    }


def _fmt(v, nd=2):
    return "—" if v is None else ("%." + str(nd) + "f") % v


def _k41(m):
    b, r2 = m["beta"], m["beta_r2"]
    if b is None or r2 is None or r2 < 0.9:
        return NOT_MEASURABLE, "β sin ley de potencia clara (R² = %s)" % _fmt(r2)
    ok = 3.3 <= b <= 4.0
    return (COMPATIBLE if ok else NOT_COMPATIBLE), "β = %.2f (esperado 3.3–4.0; 11/3 ≈ 3.67)" % b


def _supersonic(m):
    b, r2 = m["beta"], m["beta_r2"]
    if b is None or r2 is None or r2 < 0.9 or m["intermittent"] is None:
        return NOT_MEASURABLE, "falta β con ajuste claro o la curtosis por escalas"
    ok = 2.0 <= b < 3.3 and m["intermittent"]
    return (COMPATIBLE if ok else NOT_COMPATIBLE), "β = %.2f (esperado 2.0–3.3) e intermitencia %s%s" % (
        b, "por encima del nulo" if m["intermittent"] else "dentro del nulo",
        "; ojo: β ≈ 3 con intermitencia también lo da un solo borde nítido (fila de bordes)" if ok and b >= 2.7 else "")


def _edges(m):
    """Ley de Porod: superficies nítidas entre dos medios dan P(k) ∝ k^-(d+1); en una imagen 2D,
    β ≈ 3, y los saltos de brillo hacen intermitentes los incrementos. Medido con un frente nítido
    (PSF 1 px, ruido 2 %): β 3.09, intermitente, FDR 4/4 (un 🟢 sin turbulencia); 60 discos: β 3.14."""
    b, r2 = m["beta"], m["beta_r2"]
    if b is None or r2 is None or r2 < 0.9 or m["intermittent"] is None:
        return NOT_MEASURABLE, "falta β con ajuste claro o la curtosis por escalas"
    ok = 2.7 <= b <= 3.4 and m["intermittent"]
    return (COMPATIBLE if ok else NOT_COMPATIBLE), "β = %.2f (un borde nítido da ≈ 3) e intermitencia %s" % (
        b, "por encima del nulo" if m["intermittent"] else "dentro del nulo")


def _filaments(m):
    fe = m["filament_excess"]
    if fe is None:
        return NOT_MEASURABLE, "sin medida de crestas"
    return (COMPATIBLE if fe >= RIDGE_MIN else NOT_COMPATIBLE), "exceso de crestas %.4f (umbral %.3f)" % (fe, RIDGE_MIN)


def _magnetic(m):
    p, R = m["orientation_p"], m["orientation_coherence"]
    if p is None or R is None:
        return NOT_MEASURABLE, "sin prueba de dirección preferente (JSON de una versión anterior de la app: repite el análisis)"
    ok = p <= 0.01
    return (COMPATIBLE if ok else NOT_COMPATIBLE), (
        "%s: estructuras a %s°, coherencia %.2f frente a campos sin dirección (95 %% hasta %s), p = %.2f"
        % ("dirección preferente" if ok else "sin dirección preferente", _fmt(m["structure_direction"], 0), R,
           _fmt(m["orientation_null_q95"]), p))


def _shells(m):
    b1 = m["betti_1"]
    if b1 is None:
        return NOT_MEASURABLE, "sin topología"
    return (COMPATIBLE if b1 >= 1 else NOT_COMPATIBLE), "%d huecos rodeados de emisión (β1)" % int(b1)


def _fragmentation(m):
    return (COMPATIBLE if m["periodic"] else NOT_COMPATIBLE), (
        "espaciado periódico coherente y no alineado al detector" if m["periodic"] else "sin espaciado periódico fiable")


def _hierarchical(m):
    lac, mf = m["lacunarity"], m["multifractality"]
    if lac is None or mf is None:
        return NOT_MEASURABLE, "sin lacunaridad o multifractalidad"
    ok = lac >= 2.0 and mf >= 0.05
    return (COMPATIBLE if ok else NOT_COMPATIBLE), "lacunaridad %.2f (≥ 2) y multifractalidad %.3f (≥ 0.05)" % (lac, mf)


def _never(reason):
    return lambda m: (NOT_MEASURABLE, reason)


MECHANISMS = [
    {"key": "k41", "name": "Turbulencia subsónica (cascada de Kolmogorov)", "rule": _k41,
     "predicts": "Misma pendiente β en otro trazador del mismo gas; dispersión de velocidades que crece con el tamaño.",
     "question": "¿Se mantiene β ≈ 11/3 en otro filtro? ¿Las líneas espectrales dan σ ∝ L^0.38 (relación de Larson)?",
     "refs": "Kolmogorov 1941, Dokl. Akad. Nauk SSSR 30, 301; Lazarian & Pogosyan 2000, ApJ 537, 720; "
             "Larson 1981, MNRAS 194, 809"},
    {"key": "supersonic", "name": "Turbulencia supersónica / compresible (choques)", "rule": _supersonic,
     "predicts": "Espectro más plano que 11/3, saltos bruscos (intermitencia) y una distribución log-normal de densidad.",
     "question": "¿Las anchuras de línea dan número de Mach > 1? ¿La distribución de brillo es log-normal con "
                 "anchura σ² = ln(1 + b²M²)?",
     "refs": "Kim & Ryu 2005, ApJ 630, L45; Padoan, Nordlund & Jones 1997, MNRAS 288, 145; "
             "Federrath et al. 2010, A&A 512, A81"},
    {"key": "edges", "name": "Alternativa: bordes nítidos (un frente, un choque o el borde de una nube)", "rule": _edges,
     "predicts": "β ≈ 3 (ley de Porod) en cualquier filtro que vea el mismo borde y la señal a lo largo de una o pocas "
                 "líneas. Un borde nítido también aplana la turbulencia que haya debajo (medido: β 3.67 → 3.10).",
     "question": "¿El mapa local concentra la señal a lo largo del borde? ¿Cambian β y el semáforo si eliges una región "
                 "que no cruce el borde?",
     "refs": "Porod 1951, Kolloid-Zeitschrift 124, 83"},
    {"key": "filaments", "name": "Filamentos de nube molecular (cuna de estrellas)", "rule": _filaments,
     "predicts": "Anchura casi constante ≈ 0.1 pc; núcleos densos a lo largo del filamento.",
     "question": "Con la distancia al objeto: ¿la anchura del perfil transversal es ≈ 0.1 pc? ¿Hay núcleos o "
                 "protoestrellas sobre las crestas?",
     "refs": "Arzoumanian et al. 2011, A&A 529, L6"},
    {"key": "magnetic", "name": "Estructura ordenada por el campo magnético", "rule": _magnetic,
     "predicts": "Las estructuras se alinean con el campo (o perpendiculares en las zonas más densas).",
     "question": "¿La polarimetría (polvo en el IR o submilimétrico) muestra el campo paralelo o perpendicular a las "
                 "estructuras?",
     "refs": "Goldreich & Sridhar 1995, ApJ 438, 763; Planck Collaboration Int. XXXV 2016, A&A 586, A138"},
    {"key": "shells", "name": "Burbujas o cáscaras por retroalimentación estelar", "rule": _shells,
     "predicts": "Una estrella caliente o un cúmulo dentro del hueco; gas ionizado en el interior y PAH en el borde.",
     "question": "¿Hay una fuente ionizante en el centro del hueco? ¿Líneas de gas ionizado (Paα, Brγ) dentro y "
                 "emisión de PAH (3.3, 7.7 µm) en el borde?",
     "refs": "Churchwell et al. 2006, ApJ 649, 759"},
    {"key": "fragmentation", "name": "Fragmentación gravitatoria periódica", "rule": _fragmentation,
     "predicts": "Núcleos espaciados ≈ 4 veces la anchura del filamento.",
     "question": "¿El espaciado se repite en otro filtro y vale ≈ 4 × la anchura del filamento?",
     "refs": "Inutsuka & Miyama 1992, ApJ 388, 392"},
    {"key": "hierarchical", "name": "Medio jerárquico / fractal (grumos dentro de grumos)", "rule": _hierarchical,
     "predicts": "La misma estadística al cambiar de escala (autosimilaridad).",
     "question": "¿La lacunaridad y la multifractalidad se mantienen al degradar la resolución 2× y 4×?",
     "refs": "Elmegreen & Falgarone 1996, ApJ 471, 816; Chappell & Scalo 2001, ApJ 551, 712"},
    # Física básica de las nebulosas de reflexión y regiones H II (NGC 7023, Barra de Orión, Cabeza de
    # Caballo). Faltaba: en NGC 7023 la tabla podía decir "ningún mecanismo encaja" sin mencionarla.
    {"key": "pdr", "name": "Frente de fotodisociación iluminado por una estrella (PDR)",
     "rule": _never("una imagen no dice dónde está la estrella iluminadora ni separa las capas del frente"),
     "predicts": "Borde brillante orientado hacia la estrella; capas ordenadas con la distancia a ella: PAH "
                 "(7.7, 11.3 µm), H₂ (2.12 µm) y CO, desplazadas entre sí.",
     "question": "¿El borde más brillante mira hacia la estrella iluminadora (en NGC 7023, HD 200775)? ¿Los filtros de "
                 "PAH, de H₂ y del continuo del polvo muestran capas desplazadas entre sí?",
     "refs": "Tielens & Hollenbach 1985, ApJ 291, 722; Hollenbach & Tielens 1997, ARA&A 35, 179"},
    {"key": "collapse", "name": "Colapso por autogravedad",
     "rule": _never("necesita la distribución de densidad de columna, que una imagen de brillo no da"),
     "predicts": "Cola de ley de potencia en la distribución de densidad de columna.",
     "question": "Con un mapa de densidad de columna (extinción o polvo): ¿la distribución tiene cola de ley de potencia?",
     "refs": "Kainulainen et al. 2009, A&A 508, L35"},
    {"key": "galaxies", "name": "Alternativa: campo de galaxias (no es gas ni polvo)",
     "rule": _never("la imagen sola no dice si la emisión es una nube de nuestra galaxia o un campo extragaláctico"),
     "predicts": "La estructura sigue la distribución de galaxias (agrupamiento), no la del gas.",
     "question": "¿El objetivo es extragaláctico (campo profundo, cúmulo de galaxias)? Si lo es, las filas de "
                 "turbulencia, filamentos y campo magnético no aplican: mide el agrupamiento de las galaxias.",
     "refs": "Peebles 1980, The Large-Scale Structure of the Universe (Princeton University Press)"},
    {"key": "projection", "name": "Alternativa: superposición de nubes en la línea de visión",
     "rule": _never("una imagen 2D no separa capas a distinta distancia"),
     "predicts": "Varias componentes de velocidad en la misma posición.",
     "question": "¿Un espectro de líneas muestra más de una componente de velocidad en la región?",
     "refs": "—"},
    {"key": "unresolved", "name": "Alternativa: fuentes débiles no resueltas (estrellas, galaxias)",
     "rule": _never("las fuentes por debajo de 5σ no se enmascararon"),
     "predicts": "La estructura coincide con objetos de catálogos más profundos.",
     "question": "¿La estructura coincide con fuentes de un catálogo más profundo (Gaia, campos profundos del HST/JWST)?",
     "refs": "—"},
]


def build(card, level):
    """Tabla de hipótesis para el JSON y la app. `level`: nivel del semáforo."""
    if level not in ("robust", "pioneer"):
        return {"active": False, "level": level,
                "note": "Las hipótesis solo se generan con el semáforo en 🟢 o 🟣: sin estructura robusta no hay "
                        "nada que explicar."}
    m = measurements(card)
    rows = []
    for mech in MECHANISMS:
        status, why = mech["rule"](m)
        rows.append({"mechanism": mech["name"], "key": mech["key"], "status": status, "why": why,
                     "predicts": mech["predicts"], "question": mech["question"], "refs": mech["refs"]})
    compatible = [r["mechanism"] for r in rows if r["status"] == COMPATIBLE]
    if compatible:
        cls, title = "known_physics", "Patrón compatible con física conocida"
        summary = ("Compatible con: %s. Las preguntas de la tabla deciden entre ellas; ninguna está demostrada."
                   % "; ".join(compatible))
    else:
        cls, title = "open_question", "Pregunta abierta: ningún mecanismo de la tabla encaja"
        open_rows = [r["mechanism"] for r in rows if r["status"] == NOT_MEASURABLE]
        summary = ("La estructura es robusta pero no encaja con ningún mecanismo que una imagen pueda medir. %s"
                   "Antes de hablar de física nueva: confírmala en otro filtro, otra época u otro instrumento, y "
                   "descarta las alternativas (proyección, fuentes no resueltas)."
                   % ("Quedan %d sin poder medirse con una sola imagen (%s): sus preguntas van primero. "
                      % (len(open_rows), "; ".join(open_rows)) if open_rows else ""))
    target = str((card.get("source") or {}).get("target") or "").strip()
    context = ("Objeto según la cabecera FITS: %s. " % target if target else "La cabecera no indica el objeto. ") + (
        "Las hipótesis de turbulencia, filamentos, campo magnético y burbujas son de gas y polvo interestelar: solo "
        "aplican si la región es una nube o nebulosa. En un campo de galaxias, β y la anisotropía describen cómo se "
        "reparten las galaxias.")
    return {"active": True, "level": level, "classification": cls, "title": title, "summary": summary,
            "context": context, "target": target, "measurements": m, "rows": rows}
