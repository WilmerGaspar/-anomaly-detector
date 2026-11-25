# CLAUDE.md — Cosmic Materials Scout (`-anomaly-detector`)

## Cómo trabajar con el usuario

- Responder siempre en español.
- El usuario está estudiando programación: explicar en palabras simples qué se hizo y por qué.
- Revisar el código de terceros (skills, plugins, paquetes) antes de instalarlo.

## El proyecto

App de Streamlit que busca organización tipo material (agregado, filamento, red, campo crítico…)
en imágenes FITS del cielo y la compara con un nulo de fase (IAAFT). No usa deep learning y
ningún resultado afirma un descubrimiento. Ver `README.md`, `PLAN.md` (incluye la lista «No hacer»)
y `DEPLOY.md` (Streamlit Community Cloud).

## Comandos

```bash
pip install -r requirements.txt   # el contenedor de la nube no las trae instaladas (~1 min)
streamlit run app.py
python3 -m pytest -q              # 282 pruebas, ~5 min
```

Producción usa Python 3.11 (`runtime.txt`); el contenedor de la nube trae 3.13.

## Convenciones

- Mensajes de commit en español.
- Cada cambio agrega una entrada en `CHANGES.md` (qué estaba mal, cómo se verificó, qué lo prueba)
  y pruebas en `tests/`.

## Memoria entre sesiones

Al terminar una sesión con algo que valga la pena recordar, agregar aquí una entrada corta
(fecha, qué se hizo, qué quedó pendiente) y subirla con el resto de los cambios.

- **2026-11-25** — Se instalaron los skills de `academic-research-skills` (deep-research,
  academic-paper, academic-paper-reviewer, academic-pipeline, sr-screener). Quedan permanentes
  gracias al script de configuración del entorno «Default», fijado al commit `18749d8` del repo
  Imbad0202/academic-research-skills. No se instaló `claude-mem`: en la nube su memoria se borra
  con el contenedor y envía toda la actividad a un modelo observador. Este archivo hace de memoria.
