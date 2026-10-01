# Cambios — revisión de rigor (octubre 2026)

Cada cambio indica qué estaba mal, cómo se verificó y qué lo prueba ahora.
Los tests están en `tests/test_null_calibration.py`.

## Estadística (`scoring.py`)

1. **El FDR no podía pasar con la configuración por defecto.** La app usaba 12
   subrogados para los descriptores: p mínimo = 1/13 ≈ 0.077, por encima del
   umbral de Benjamini-Hochberg más laxo (0.05) con 4 descriptores. Toda imagen
   salía "no candidato". Ahora el valor por defecto es 99 subrogados (p ≥ 0.01) y
   `fdr_decision` informa `p_min_possible` y `can_pass`.

2. **Fórmula del p-valor.** Antes: `max(mean(nulo >= obs), 1/(n+1))`, que
   subestima p. Ahora: `(1 + #{nulo >= obs}) / (n + 1)` (North et al. 2002,
   Am. J. Hum. Genet. 71:439). Función `mc_p_value`.

3. **Simetría hermítica.** Las fases uniformes independientes en la mitad
   `rfft2` no respetan la simetría que `irfft2` necesita. Medido: error relativo
   del espectro de 3.5 % en un campo de 96×96. Ahora las fases se toman de la
   FFT de ruido blanco real y el espectro se conserva a precisión de máquina
   (test `test_phase_surrogate_preserves_spectrum_exactly`). Se quitó además el
   `np.clip` final, que también alteraba el espectro.

4. **Nulo IAAFT por defecto** (Schreiber & Schmitz 1996, PRL 77:635). El nulo
   solo de fase vuelve gaussiano el histograma. Como la imagen llega recortada
   al percentil 2–98, cualquier estadístico sensible al histograma rechazaba
   el nulo sin que hubiera estructura espacial. IAAFT conserva el histograma
   exacto (test `test_iaaft_preserves_histogram_exactly`) y el espectro de forma
   aproximada (error relativo medido ≈ 0.08 %).

   **Medición decisiva.** Con el método original (nulo de fase + entropía y
   energía en el FDR, normalización 2–98) y 99 subrogados para que el FDR
   pudiera pasar, 15 de 15 campos exp(campo gaussiano) — sin ninguna estructura
   más allá de espectro e histograma — salían como candidatos. El defecto 1
   (FDR imposible) ocultaba este defecto.

   **Calibración del método nuevo** (99 subrogados IAAFT, BH α = 0.05):
   - campos gaussianos con β ∈ [2, 3.5]: 1 de 60 marcados (1.7 %);
   - exp(campo gaussiano), la hipótesis nula exacta de IAAFT: 1 de 30 (3.3 %);
   - control positivo, filamentos finos sobre fondo gaussiano: 10 de 10 detectados.

5. **Descriptores del FDR.** Se quitaron `entropy` (depende solo del
   histograma: idéntico bajo IAAFT) y `energy_mean` (la media del gradiente al
   cuadrado está fijada por el espectro: idéntica bajo cualquier subrogado de
   fase). Ambos hacían tests degenerados. Ahora: `aniso` (coeficiente de
   variación de la energía del gradiente), `flatness_lag1`, `flatness_lag4`
   (curtosis de incrementos a 1 y 4 px, ambos ejes) e `incr_skew` (asimetría de
   incrementos). Los cuatro dependen de las fases.

6. **Misma resolución en observado y nulo.** Antes el score observado se
   calculaba a resolución completa y los nulos a 96 px. Medido en campos
   gaussianos: el efecto es pequeño y sin sesgo sistemático, pero era una
   inconsistencia metodológica. Ahora el observado se recalcula sobre la misma
   imagen reducida. El argumento `observed_score` se conserva por compatibilidad
   y se guarda como `observed_score_full_res`.

7. **Lacunaridad sin bloques parciales.** Los bloques del borde eran más
   pequeños e inflaban la varianza de masas. Ahora la imagen se recorta a un
   múltiplo del tamaño de bloque.

8. `compute_global_score` ya no modifica el score según un "modo" (multiplicar
   por 1.15 o 0.85 no tenía justificación); devuelve NaN si no hay descriptores
   en lugar de un 0.5 inventado.

## Familias (`materials_map.py`)

9. **Descriptores ausentes ya no se inventan.** Si un plugin falla o está
   apagado, antes se imputaba un valor por defecto. `d0 = 1.5` cae dentro de la
   banda de "aggregate" y le regalaba 0.35 puntos. Ahora cada familia promedia
   solo sus descriptores disponibles y su puntuación se pondera por la fracción
   de descriptores presentes (`family_coverage`, visible en la interfaz). El z
   del nulo, común a todas las familias, no cuenta como cobertura propia.

10. La app ahora pasa `metadata` a `interpret()`; antes la advertencia de
    filtro CLEAR nunca se evaluaba.

## Datos (`mast_client.py`)

11. **Conteos reales.** `search_observations_detailed` informa cuántas
    observaciones devolvió MAST, cuántas son de la misión, cuántas son imágenes
    y cuántas se muestran. Antes se cortaba en 40 sin avisar.
12. **Procedencia por fila:** programa, PI, título, fecha (de `t_min`, MJD),
    exposición, coordenadas, nivel de calibración, derechos de datos y URLs
    devueltas por MAST. Texto de reconocimiento a MAST en `MAST_ACK`.
13. **x1d y s3d ya no se recomiendan:** son una tabla espectral y un cubo 3D,
    no imágenes. `is_image_product` decide qué es analizable; excluye
    explícitamente `uncal`, que termina en `cal.fits`.
14. **Descarga segura.** El tamaño se verifica antes de descargar. Si la
    descarga falla, se lanza un error; antes se tomaba cualquier FITS que
    hubiera en `/tmp`, que podía ser de otro objeto.
15. `download_url`: carga desde URL directa con tope de tamaño (antes el campo
    "URL FITS directo" existía en la interfaz pero no descargaba nada).
16. Se eliminó `list_mission_targets` (no se usaba y lanzaba una consulta sin
    límite a todo el archivo). Si otro módulo la importa, hay que restaurarla.

## Interfaz (`app.py`)

17. Flujo único en cuatro pasos: fuente de datos → imagen y región →
    sonificación → análisis. Selección por clic en las tablas
    (`st.dataframe(on_select=...)`, requiere **streamlit >= 1.35**).
18. Tres orígenes explícitos: MAST, archivo propio, URL directa. La procedencia
    se muestra antes de cargar (MAST) y después de cargar (cabecera FITS) y se
    guarda en el JSON.
19. **Región de estudio.** La imagen completa se muestra con un recuadro; el
    análisis corre solo sobre la región elegida (máx. 2048 px). Antes los NaN de
    borde de los mosaicos se convertían en 0 y creaban un escalón artificial.
    Ahora la normalización usa solo píxeles finitos y se avisa si más del 5 %
    de la región está vacía.
20. Se elige la extensión FITS explícitamente (SCI por defecto) y se avisa si
    se analiza ERR/WHT/VAR. Antes, sin SCI, se tomaba la última HDU con datos.
21. **Sonificación comparativa**, rotulada como sonificación: región real
    contra un subrogado IAAFT con la misma escala de tono y volumen.
22. Controles eliminados o reubicados: "MODO" (no se usaba), "ISOFOREST" (solo
    ocultaba una métrica), "URL FITS DIRECTO" (ahora funciona, en el paso 1).
    Rosenstein y Fibonacci/phi pasan a "Exploratorios", apagados por defecto.
23. Los descriptores que fallan se muestran como excluidos, no se ocultan.
24. Semilla fija (80) registrada en el JSON para reproducibilidad.

## Pendiente (no revisado en esta ronda)

`plugins/`, `nos_morphological.py`, `nos_empirical.py`, `provenance.py`,
`candidate_export.py`, `report_generator.py`, `local_rarity.py`,
`instrument_mask.py`, `fits_quality.py`, `reproducibility.py` y los tests
existentes. La app conserva las mismas llamadas a esos módulos.

## Panel analítico (`analytics.py`, `results_panel.py`)

25. El paso 4 se reorganiza en pestañas; cada número aparece junto a su nulo:
    - **Nulo y FDR:** tabla por descriptor (observado, media y banda 5–95 % del
      nulo, z, p, umbral BH, qué mide), histograma de la distribución nula de
      cada descriptor con el valor observado, gráfico de Benjamini-Hochberg y
      distribución nula del score global.
    - **Espectro:** P(k) radial con ventana de Hann, ajuste k^-β con R² en
      k = 0.02–0.25 y curva del subrogado (si no se solapa, el nulo no es válido).
    - **Escalas:** curtosis de incrementos a 1–16 px frente a la banda de 19
      subrogados IAAFT, e histograma log de incrementos (región, subrogado, gaussiana).
    - **Mapa local:** z de `flatness_lag1` en 4×4 teselas, cada una contra sus
      propios subrogados. Exploratorio, sin corrección múltiple.
    - **Familias:** puntuación con cobertura y margen entre las dos primeras.
    - **Descriptores:** tabla completa de valores numéricos de todos los plugins.
    - **Región vs nulo:** región, subrogado IAAFT y |∇| lado a lado.
26. `cheap_descriptor_null_details` (en `scoring.py`) devuelve la distribución
    nula completa; `cheap_descriptor_pvalues` lo usa y da los mismos p.
27. Exportación: el JSON incluye `analytics` y un resumen de nulos por
    descriptor; hay además CSV de tests y CSV de descriptores.
28. `requirements.txt`: `streamlit>=1.35` (la app ya lo exigía).
29. Tests nuevos en `tests/test_analytics.py` (β recuperado en campo gaussiano,
    filamentos detectados por escala, coherencia de p, mapa local, utilidades).

**Observación pendiente de revisar.** Los subrogados IAAFT de un campo
gaussiano tienen curtosis de incrementos a 1 px ≈ 3.9 en lugar de 3.0. El test
es de una cola, así que no genera falsos positivos, pero resta potencia a
`flatness_lag1`: una intermitencia moderada puede quedar dentro del nulo.
