"""
Generador de informes simples para Anomaly Detector.
Contiene `generate_report(results_dict)` que recibe el diccionario de
resultados y devuelve un informe en texto formateado con secciones:
- Resumen Ejecutivo
- Hallazgos Clave
- Advertencias
- Conclusión

El contenido es intencionadamente legible en Markdown para mostrarse
con `st.markdown` en la app.
"""
from datetime import datetime


def _safe_fmt(val, precision=3):
    try:
        if isinstance(val, float):
            return f"{val:.{precision}f}"
        return str(val)
    except Exception:
        return str(val)


def generate_report(results: dict) -> str:
    """Genera un informe en Markdown a partir de `results`.

    Se espera que `results` contenga al menos: timestamp, filename, mode,
    global_score, monte_carlo (con p_value, null_mean, null_se, null_ci95),
    y plugin_results.
    """
    ts = results.get('timestamp', datetime.now().isoformat())
    filename = results.get('filename', 'desconocido')
    mode = results.get('mode', 'desconocido')
    global_score = results.get('global_score', None)
    mc = results.get('monte_carlo', {})
    plugins = results.get('plugin_results', {})

    score_pct = _safe_fmt(global_score * 100) if global_score is not None else 'N/A'
    p_value = _safe_fmt(mc.get('p_value', 'N/A'))
    null_mean = _safe_fmt(mc.get('null_mean', 'N/A'))
    null_se = _safe_fmt(mc.get('null_se', 'N/A'))
    ci95 = mc.get('null_ci95', None)
    ci_text = f"[{_safe_fmt(ci95[0])}, {_safe_fmt(ci95[1])}]" if ci95 else 'N/A'

    # Hallazgos clave: resumir valores principales de plugins
    findings = []
    for name, res in plugins.items():
        try:
            if name == 'fractal_base':
                findings.append(f"Dimensión fractal D0={_safe_fmt(res.get('d0'))}, complejidad={_safe_fmt(res.get('complexity_score'))}")
            elif name == 'kolmogorov_1941':
                findings.append(f"Kolmogorov β={_safe_fmt(res.get('beta'))}, intermittency={_safe_fmt(res.get('intermittency_factor'))}")
            elif name == 'lyapunov_stability':
                findings.append(f"Lyapunov max={_safe_fmt(res.get('max_lyapunov'))}, tipo={res.get('dynamics_type','-')}")
            elif name == 'persistent_homology':
                findings.append(f"Betti0={res.get('betti_0',0)}, Betti1={res.get('betti_1',0)}")
            elif name == 'renormalization_group':
                findings.append(f"Longitud de correlación={_safe_fmt(res.get('correlation_length'))}, crítico={res.get('is_critical')}")
            else:
                findings.append(f"{name}: {', '.join([f'{k}={_safe_fmt(v)}' for k,v in (res or {}).items()][:3])}")
        except Exception:
            continue

    # Advertencias
    warnings = []
    if mc.get('n_simulations', 0) < 1000:
        warnings.append('Número bajo de simulaciones Monte Carlo (<1000) — intervalos poco fiables')
    if global_score is None:
        warnings.append('Score global no calculado')
    if not plugins:
        warnings.append('No se analizaron plugins — resultados parciales')

    # Construir informe en Markdown
    md = []
    md.append(f"## Resumen Ejecutivo\n")
    md.append(f"**Archivo:** {filename}  ")
    md.append(f"**Timestamp:** {ts}  ")
    md.append(f"**Modo:** {mode}  ")
    md.append(f"**Score global:** {score_pct}%  ")
    md.append(f"**Significancia (p):** {p_value} — IC95% media nula: {ci_text} — SE: {null_se}  \n")

    md.append("## Hallazgos Clave\n")
    if findings:
        for f in findings:
            md.append(f"- {f}  \n")
    else:
        md.append("- No se detectaron hallazgos cuantificables.  \n")

    md.append("## Advertencias\n")
    if warnings:
        for w in warnings:
            md.append(f"- {w}  \n")
    else:
        md.append("- Ninguna advertencia relevante.  \n")

    md.append("## Conclusión\n")
    if mc.get('is_significant'):
        md.append("- El análisis sugiere la presencia de una anomalía estadísticamente significativa. Se recomienda una revisión más profunda y, si procede, validación con datos independientes.  \n")
    else:
        md.append("- No se encontraron evidencias estadísticas suficientes para catalogar la imagen como anómala.  \n")

    md.append("---\n")
    md.append("_Generado por Anomaly Detector_\n")

    return '\n'.join(md)
