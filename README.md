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
