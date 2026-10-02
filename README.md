# Cosmic Materials Scout

Herramienta para buscar organizacion tipo material en imagenes del universo (FITS / campo 2D).

No identifica un compuesto y no usa deep learning. Extrae descriptores fisicos,
los compara con un nulo que conserva el espectro y baraja la fase, y propone
un analogo de laboratorio.

Repo: WilmerGaspar/-anomaly-detector

## Pregunta

En este recorte del cielo, hay una organizacion espacial que en fisica de
materiales llamariamos agregado, filamento, red, condensado o campo casi
critico — y que no se explica solo con ruido del mismo color?

Identificar la sustancia exige espectro y contexto del instrumento.

## Familias

| Familia | Cielo | Laboratorio |
| --- | --- | --- |
| aggregate | Polvo autosimilar | Hollin, DLA, aerogeles |
| cascade | ISM turbulento | Medio intermitente |
| filament | Filamentos magnetizados | Polimeros, texturas |
| lattice | Picos discretos en FFT | Cristal / superred o fringing CCD |
| compact | Nudo / estrella | Grano, nucleacion |
| critical | Scale-free | Percolacion |
| featureless | Fondo | Vidrio / ruido |

## Que calcula

- Fractal: box-counting D0, Dq por masas, lacunaridad
- Espectro: beta de P(k), flatness de incrementos, isotropia
- Periodicidad: picos sobre el continuo radial del |FFT|^2 + aviso si alinean al detector
- Anisotropia: tensor de estructura
- Topologia: componentes y agujeros por umbral
- Coarse-graining 2x2
- Rosenstein sobre media por filas (proxy espacial)
- Nulo: subrogados de fase

## Alerta de descubrimiento (semáforo)

Cada análisis termina en un nivel: 🔴 no válido, ⚪ sin estructura, 🟠 sin confirmar,
🟡 explicado por un confusor (bordes vacíos, fuentes puntuales, picos de difracción,
artefactos, dependencia de la semilla), 🟢 estructura robusta o 🟣 alerta de pionero
(robusta y atípica frente a tus análisis previos). Ningún nivel afirma un
descubrimiento: el más alto pide revisión experta y datos independientes.

## Ventana Fibonacci / φ

Ángulo áureo entre fuentes y razones φ entre escalas, cada uno con su nulo. Alerta
solo con p < 0.005. Si la resolución no permite distinguir φ de 3/2, lo dice.

## Señales de radio

Modo "📡 Señales de radio": biblioteca de firmas (púlsar, ráfaga dispersada,
portadora con deriva, RFI) y detectores calibrados contra su nulo. DM ≈ 0 y deriva
≈ 0 se marcan como probable interferencia terrestre. Comparación con catálogos
públicos exportados en CSV (ATNF psrcat, CHIME/FRB).

## Instalacion

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Flujo con el resto del lab

1. Recorte FITS (nube, filamento, resto de supernova, disco).
2. Si sale candidato aggregate / filament / critical / lattice (y lattice no es artefacto): guardar informe.
3. Espectro de esa region.
4. Raman/FTIR de un analogo: spectral-identifier-v1
5. Formula candidata: nos-calculator

## Limites

- Un JPEG de Hubble con spikes no es un cuasicristal.
- El nulo no sabe si el objeto es raro en astronomia; solo si tiene coherencia de fase.
- Sin corpus etiquetado no hay tasa de falsos positivos de dominio.

## Autor

Wilmer Gaspar Espinoza Castillo
