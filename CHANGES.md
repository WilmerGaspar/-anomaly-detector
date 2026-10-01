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

## Correcciones medidas (periodicidad, filamentos, sensibilidad del nulo)

Controles sintéticos usados: campos gaussianos β ∈ {2, 2.8, 3.5}, red
hexagonal, rejilla girada, fringing alineado al eje, filamentos rectos
(cruzados y paralelos), arcos curvos y manchas compactas. Los tests están en
`tests/test_periodicity_ridges.py`.

30. **`periodicity`: el ruido se marcaba como red.** El umbral fijo de exceso
    (8) quedaba por debajo de la mediana del máximo en ruido puro (≈ 9 en
    imágenes de 128 a 400 px). Un campo gaussiano daba `periodicity_score` 0.48.
    Ahora el umbral es ln(N/0.01), Bonferroni al 1 % sobre los píxeles
    independientes del espectro (≈ 13.9 con 256 px). Ruido: 0 picos en 90 campos
    de 128 a 400 px. Con 800 px, 1 pico en el 10 % de los campos (score medio 0.02).
31. **`periodicity`: picos perdidos.** El bucle solo examinaba 1 de cada 2–3
    píxeles, y un pico cuyo máximo caía en un píxel saltado desaparecía. Una
    rejilla girada de amplitud 0.8σ daba score 0 con el código original; ahora
    da 0.99. La búsqueda es vectorizada sobre todos los píxeles.
32. **`periodicity`: las líneas rectas contaban como red.** Una línea deja
    en la FFT una raya de picos con el mismo ángulo a frecuencias no armónicas.
    Esos grupos (≥ 3 picos, ±6°, no armónicos) se separan como `streak_angles_deg`
    y no puntúan como periodicidad. Una franja de fringing con un solo pico
    significativo alineado al eje ya basta para el aviso de artefacto.
33. **Nuevo `plugins/ridges.py` y familia `filament`.** La familia dependía
    del índice de anisotropía, que es bajo cuando los filamentos tienen varias
    orientaciones. El plugin mide crestas brillantes (hessiano, σ = 1 y 2 px)
    con el umbral fijado en el percentil 95 de 3 subrogados IAAFT, y cuenta los
    píxeles de esqueletos de al menos 12 px. Exceso medido: ruido, red y manchas
    ≤ 0.001; filamentos rectos o curvos 0.011–0.030. `filament` usa ahora
    max(crestas, rayas FFT) con peso 0.55. Resultado en 9 imágenes con
    filamentos: `filament` queda primera en las 9 (antes ganaba `lattice`).
    El margen sigue por debajo de 0.05, así que la app dice "Mezcla": el fondo
    turbulento puntúa legítimamente como cascade/aggregate. La red hexagonal
    sigue en `lattice` y el ruido y las manchas no dan `filament`.
34. **Sensibilidad del nulo IAAFT (no es un fallo de código).** Con espectros
    empinados, la reordenación por rangos de IAAFT da a los subrogados más
    curtosis de incrementos que el campo real: 2.99 con β = 2, 3.24 con β = 2.8
    y 4.38 con β = 3.5, frente a 3.0 con subrogados de fase. No depende del
    número de iteraciones, del punto de arranque ni de devolver el paso de
    espectro exacto (se midieron las tres variantes). El test sigue sin dar
    falsos positivos (es de una cola), pero pierde potencia. No se cambia el
    nulo: el de fase rompe la calibración con histogramas recortados (punto 4).
    La app avisa cuando β > 2.5.

## Bordes vacíos de mosaico (detectado con un resultado real de MIRI)

Un análisis real (`jw01192006001_0310h_00001_mirimage_i2d`, región x=15, y=8,
1006 px) salió `needs_spectrum` con 3 de 4 descriptores en el FDR. Dos teselas
del mapa local eran NaN, lo que solo ocurre cuando la tesela entera es borde
vacío rellenado con la mediana: al menos el 12 % de la región no tenía dato.

35. **Medido:** campos gaussianos sin estructura más una cuña vacía rellenada con
    la mediana (igual que hace la app). El FDR pasa en 0/6 campos con 0 % de
    vacío, 1/6 con 5 %, 2/6 con 12 % y 4/6 con 20 %; con 1–2 % pasa en 1/10, y
    con 0.5 % en 0/10. El escalón del borde se lee como estructura.
36. **Píxeles malos sueltos no son el problema:** con 0.2 % y 1 % de NaN
    aislados, 0/8 campos pasan, tanto con relleno por mediana global como local.
37. **Región por defecto = mayor cuadrado sin zonas vacías** (`largest_finite_square`,
    búsqueda binaria con imagen integral; 0.9 s en 4200×4200). Las "zonas vacías"
    son componentes de NaN de ≥ 64 píxeles (`large_holes`); los sueltos no cuentan.
38. **Región con > 0.1 % de zona vacía = `invalid_region`:** nunca es candidata,
    el veredicto empieza con el porcentaje y la app lo muestra en rojo antes y
    después de analizar.
39. El mapa local deja en NaN las teselas con zona vacía en vez de evaluarlas.
40. El JSON incluye un bloque `analysis` con región, fracción de NaN, fracción de
    zona vacía, normalización, nulo (método, n, semilla), descriptores activos y
    extensión. Antes la semilla no llegaba al JSON (el punto 24 lo daba por hecho).
