"""Aspecto Windows 2000 (theme.py) y consola de las IA (ai_panel): el texto de las IA nunca entra como HTML,
cada dato citado se marca ✓/✗ y la barra de tareas solo enlaza secciones que están en pantalla."""
import json
import re

import ai_hub as H
import ai_panel as P
import theme

CTX = json.dumps({"ANALISIS": {"medidas": {"beta_espectro": 2.6}, "semaforo": {"nivel": "robust"}}})


def test_ai_text_is_escaped_and_citations_are_marked():
    known = H.known_names(CTX)
    out = P._answer_html("**Qué muestra:** β = 2,6 [beta_espectro] en [mision_JWST] <script>alert(1)</script> "
                         "<img src=x onerror=alert(2)> ![p](https://x.org/p.png)", known)
    assert "<script>" not in out and "<img" not in out and "&lt;script&gt;" in out       # texto, no código
    assert '<span class="cite ok">beta_espectro ✓</span>' in out
    assert '<span class="cite bad">mision_JWST ✗</span>' in out
    assert "![p]" not in out                                                           # las imágenes no se cargan
    assert "\n\n**Qué muestra:**" in "\n\n" + out or out.startswith("**Qué muestra:**")
    # Los enlaces [texto](url) no son citas.
    assert "cite" not in P._answer_html("Más en la [web](https://example.org).", known)


def test_check_row_follows_the_verifier():
    row = P._checks_html(["cita datos que no existen: [mision_JWST]", "afirma de más («sin duda»)"])
    assert '<span class="y">✓ números</span>' in row and '<span class="n">✗ datos citados</span>' in row
    assert '<span class="n">✗ sin afirmar de más</span>' in row and '<span class="y">✓ referencias</span>' in row


def test_cite_ok_matches_verify():
    known = H.known_names(CTX)
    assert H.cite_ok("medidas.beta_espectro", known) and H.cite_ok("semaforo", known)
    assert not H.cite_ok("objeto_NGC_7023", known) and H.cite_ok("lo_que_sea", None)
    assert H.known_names("texto sin JSON") is None


def test_taskbar_links_only_to_sections_on_screen():
    assert [h for _, h in theme.tasks_for(False)] == ["#guia", "#fuente"]
    assert [h for _, h in theme.tasks_for(True)] == ["#guia", "#fuente", "#region", "#analizar"]
    from pathlib import Path
    anchors = set(re.findall(r'anchor="([a-z]+)"', (Path(__file__).resolve().parents[1] / "app.py").read_text()))
    assert {h[1:] for _, h in theme.tasks_for(True)} <= anchors, anchors           # cada acceso tiene su sección


def test_theme_loads_nothing_from_outside():
    urls = re.findall(r"url\(\"?([^\")]+)", theme.CSS)
    assert urls and all(u.startswith("data:image/svg+xml") for u in urls)
    assert "@import" not in theme.CSS and "http" not in theme.CSS.replace("http://www.w3.org/2000/svg", "")
