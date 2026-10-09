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

## Memoria (la app se caía en Streamlit Cloud tras el punto 37)

Pico de memoria del proceso de Streamlit con un FITS de 4236×4214 (68 MB),
región por defecto y análisis completo:

| Versión | Pico |
| --- | --- |
| Antes de los puntos 35–40 | 632 MB |
| Con los puntos 35–40 | 906 MB |
| Con estos cambios | 638 MB |

41. La búsqueda de la región limpia trabajaba a resolución completa (pico de
    534 MB, imagen integral int64). Ahora usa `HoleMap`: bloques f×f con
    mapa ≤ 1024 px de lado, imagen integral int32 y dilatación de 1 bloque para no
    dejar borde parcial dentro. Pico: 30 MB. El cuadrado sigue sin ningún píxel
    de zona vacía (tests con mosaicos de 400 a 4236 px).
42. La región por defecto vuelve a tener como máximo 1024 px (ahora centrada
    en la zona limpia). Con 2025 px, `kolmogorov_1941` sumaba 239 MB.
43. `list_image_hdus` lee el tamaño de la cabecera (NAXISn) en lugar de cargar
    los datos de cada extensión (+135 MB con 4200 px).

## Tipo de producto JWST (resultado real de NIRSpec)

Un análisis de `jw01192008001_02120_00001_nrs1_cal.fits` (NIRSpec, F110W) salió
`exploratory_only`, lo cual es correcto, pero con dos fallos.

44. El veredicto decía "Producto de detector (uncal/rate)", un texto fijo y
    falso para un `_cal`. Ahora dice el motivo real tomado de la procedencia
    (detector, plano de espectrógrafo o imagen de adquisición).
45. La procedencia usa `EXP_TYPE` de la cabecera cuando existe: espectroscopía
    (NRS_MSASPEC, NRS_FIXEDSLIT, NRS_IFU, MIR_MRS, NIS_SOSS, NRC_WFSS…) o
    adquisición (NRS_MSATA, NRS_TACQ, NRS_WATA…). Sin `EXP_TYPE` se mantiene la
    deducción por el nombre. La app muestra el tipo de exposición.
46. La lista de MAST mostraba los `_nrs1/_nrs2_cal` y los `mirifushort/long`
    como "imagen 2D analizable". Ahora no lo son, y la nota dice "plano de
    espectrógrafo o adquisición (no imagen del cielo)". Siguen visibles
    desactivando el filtro.

## Fuentes puntuales y picos de difracción (resultado real de MIRI F2100W)

`jw01192-o006_t013_miri_f2100w_i2d` (región limpia, 4/4 descriptores en el FDR)
salió `needs_spectrum` con `filament` primera. Pero las crestas estaban a nivel
de ruido (exceso 0.0017): `filament` ganaba solo por una "raya" en la FFT (113°),
compatible con un pico de difracción. Además: β0 = 14, β1 = 0 (manchas sueltas,
sin red) y curtosis k62 = 896, lo típico de fuentes puntuales.

47. **Medido:** ruido solo → 0 descriptores en el FDR (6/6 campos); el mismo ruido
    con 14 fuentes puntuales → de 1 a 4. Las fuentes puntuales bastan para pasar
    el FDR.
48. **Las rayas de la FFT ya no dan puntos a `filament`.** Un solo pico de
    difracción bastaba para que ganara.
49. **`ridges` separa los picos de difracción:** son crestas que salen
    radialmente (≥ 60 % del esqueleto en ≤ 8 direcciones de 5°) de una fuente
    puntual al menos 5 veces más brillante que la cresta, en valores lineales.
    La app pasa la región sin estirar (`raw`), porque al recortar a [0, 1] una
    estrella brillante satura y se funde con sus picos. Medido en 30 campos:
    filamentos (rectos, curvos, cruzados, con estrellas encima) 0.011–0.030;
    ruido, redes, manchas y estrellas con picos ≤ 0.0006.
50. **Fuentes puntuales (`compact_sources_linear`):** máximo local redondo
    (l2/l1 ≥ 0.5) más de 10 veces el ruido (MAD) sobre el fondo. Detecta 0 en
    ruido, filamentos, redes y manchas extensas (12 campos) y de 4 a 11 en
    campos de estrellas. Con 3 o más fuentes y FDR que pasa, el veredicto y la
    app avisan de que el FDR puede deberse a ellas.
51. **`periodicity`: picos sueltos a radios distintos no son una red.** Dos
    filamentos rectos dejaban picos en dos direcciones (radios 115 y 81) que
    contaban como `lattice`. Ahora hace falta la misma frecuencia en direcciones
    distintas, o armónicos en una (`lattice_consistent`).

Resultado en los controles: `filament` en 12/12 campos con filamentos, `lattice`
en 3/3 redes, y ningún campo de estrellas sale `filament`. No hay una familia
para "campo de fuentes puntuales": esos campos quedan en cascade/aggregate con
el aviso.

## Alerta de descubrimiento, ventana φ y señales de radio

Origen: un análisis real de MIRI (`jw01192006001_03105_00002_mirimage_cal`, F1000W)
con 93 fuentes puntuales salía `needs_spectrum` y candidato, aunque las fuentes
puntuales por sí solas bastan para pasar el FDR (punto 47).

### Semáforo (`discovery.py`)

52. Niveles: 🔴 no válido · ⚪ sin estructura · 🟠 sin confirmar (faltan controles)
    · 🟡 explicado por un confusor · 🟢 estructura robusta · 🟣 alerta de pionero
    (robusta y atípica frente a tus análisis previos). **Una comprobación que no se
    pudo hacer cuenta como no superada**: sin esta regla, el JSON de MIRI salía 🟢.
53. **Fuentes puntuales enmascaradas.** Cada fuente se sustituye por píxeles al
    azar de su anillo (un valor constante crearía mesetas) y se repite el FDR.
    Medido: ruido + 40 estrellas pasaba hasta 3/4 → 0/4 tras enmascarar (4/4
    campos). Filamentos + estrellas tras enmascarar = filamentos solos (0, 0, 3,
    1 de 4 en ambos casos): la prueba quita las estrellas sin borrar estructura.
54. Réplica con otra semilla, picos de difracción, artefactos, zonas vacías y
    procedencia son parte de las comprobaciones. `is_candidate` en el JSON sigue
    al semáforo (solo 🟢 o 🟣).
55. Novedad: z robusto (mediana, 1.4826·MAD) por descriptor frente a ≥ 5 JSON
    previos que subes tú. |z| ≥ 5 con estructura robusta → 🟣. Dice "distinto de lo
    que ya analizaste", no "nuevo para la ciencia".

### Ventana Fibonacci / φ (`golden.py`)

56. Ángulo áureo (137.508°) entre fuentes ordenadas por radio; nulo: barajar el
    orden. Espiral de Vogel p ≤ 0.004; puntos al azar p = 0.12–0.71; girasol
    simulado detectado desde la imagen p = 0.001.
57. Razones φ y φ² entre picos del espectro radial; nulo: mismo número de picos
    en frecuencias al azar. **Primera versión descartada:** a la resolución usada,
    anillos con razón 3/2 daban el mismo "resultado φ" que los de razón φ. Ahora
    solo cuentan pares resolubles a ±5 % y el espectro usa hasta 1024 px. Medido
    (1024 px): φ p = 0.001; 3/2 p = 0.41–0.46; ruido p = 0.83–1.0. Si no hay
    resolución, dice "resolución insuficiente". Alerta solo con p < 0.005.

### Señales de radio (`radio_signals.py`, `radio_page.py`)

58. Biblioteca de firmas generadas por su ecuación: púlsar, ráfaga dispersada
    (t ∝ DM·f⁻², K = 4.148808 ms GHz² pc⁻¹ cm³), portadora con deriva Doppler, y dos
    RFI (impulso con DM = 0, portadora con deriva 0). Enlaces a ATNF psrcat,
    CHIME/FRB y Breakthrough Listen Open Data (la app no se conecta a ellos).
59. Calibración medida (y fallos corregidos por el camino):
    - Periodicidad: falsas alarmas en ruido blanco 1.2 % (nominal 1 %), ruido
      rojo 1/100, púlsar 20/20. Errores encontrados: relleno del filtro de
      mediana (48 % de falsas alarmas), ventana fija con ruido rojo (30/30),
      primer bin de frecuencia (10/30), índices de armónicos tras cortar
      frecuencias bajas (sensibilidad 6/20 → 20/20).
    - Periodo afinado por plegado con incertidumbre: error real ≤ 0.33 σ en 20
      casos. El cruce con catálogo usa esa incertidumbre (antes no encontraba un
      púlsar medido con 0.3 % de error).
    - Dispersión: ruido 2/110 (1.8 %), ráfagas DM 500 detectadas 5/5 con DM 497–503,
      ruido de colas pesadas 0/10, impulso RFI con DM = 0 marcado.
    - Deriva: ruido 0/20, derivas −0.15 y 0.37 Hz/s exactas 5/5, deriva 0 marcada
      como RFI. Rejilla de derivas con la resolución real (con pasos de 0.1 Hz/s,
      −0.15 Hz/s no se detectaba).
60. Entradas: serie temporal (CSV/TXT/NPY/FITS), espectro dinámico canales×tiempo,
    espectrograma tiempo×canales. Los formatos de radiotelescopio (.fil, .h5)
    deben convertirse antes.

## Prueba de enmascarado: una sola estrella y sus picos (MIRI F2100W real)

`jw01192-o006_t013_miri_f2100w_i2d` (FDR 4/4, 1 fuente brillante, 2 picos de
difracción) salió 🟡 sin haberlo demostrado: la prueba de enmascarar solo se hacía
con ≥ 3 fuentes y los picos contaban como fallo solo por existir.

61. La prueba de enmascarado se ejecuta **siempre**. Enmascara desde 5σ (a 10σ,
    estrellas débiles mantenían el FDR en 1 de 4 campos de estrellas con picos).
62. Los picos de difracción se tapan como **bandas finas** (±3 px) a lo largo de
    su cresta. Primera versión descartada: discos que contenían los picos tapaban
    el 60–95 % de la imagen.
63. Relleno = interpolación suave desde los vecinos + ruido del nivel local. El
    relleno con píxeles al azar rompía la correlación del fondo en zonas grandes.
64. Un pico de difracción debe además **apagarse con la distancia** (brillo a
    R–2R ≥ 1.3 × brillo a 3R–6R). Sin esto, tramos de filamento junto a estrellas
    se tomaban por picos y se tapaban.
65. El control de picos solo falla si la señal desaparece al taparlos.
66. Resultado en 28 campos simulados (7 tipos × 4 semillas):
    - ruido + estrellas (1 o 40) y estrellas con picos: la señal desaparece al
      enmascarar en 12/12 → 🟡 con el motivo demostrado;
    - ruido solo y filamentos solos: el enmascarado no cambia nada;
    - filamentos + 1 estrella: la estructura sobrevive en 4/4;
    - filamentos + 40 estrellas: se pierde en 1–2 de 4 (error prudente);
    - ningún 🟢 falso.

## Falso 🟢 y falsa alerta φ en un MIRI F2550W real

`jw01192006001_0310h_00003_mirimage_i2d` salió 🟢 "robusta" con alerta de ángulo
áureo. Ambos eran falsos.

67. **Bug matemático en el ángulo áureo.** Se usaba |media exp(i(Δθ − 137.5°))|,
    cuyo módulo no depende del desfase: cualquier concentración de giros daba
    alerta (puntos agrupados en una nube → p = 0.001). Ahora: media de cos(Δθ ∓
    137.5°). Medido: nubes simuladas p = 0.87–1.0; espiral de Vogel sigue p ≤ 0.004.
68. **"1163 fuentes puntuales" que eran emisión extendida** (radio en el tope,
    38 px). Las fuentes se miden ahora frente al fondo local (mediana en bloques de
    32 px) y deben tener la luz concentrada como una PSF: (media r ≤ 2s − pedestal)
    / (media 2s–4s − pedestal) ≥ 5 en alguna escala s, con pedestal = mediana a
    4s–6s. Gaussiana ≈ 20, grumo extendido ≈ 1.5. Estrellas: 30/30 y 40/40
    detectadas; ruido: 0.
69. Una fuente cuyo perfil no baja al fondo local no se enmascara (es extendida).
70. **La prueba de enmascarado no es válida si cubre más del 10 % del área** → la
    comprobación queda "no comprobado" (🟠 como máximo). Medido: con discos al azar
    el relleno no fabrica estructura ni al 15 % (0/18 por tipo de campo), pero tapar
    selectivamente los grumos brillantes de una emisión extendida sí: FDR 0/4 →
    3–4/4. Sin un modelo de PSF del instrumento, un nudo compacto de polvo y una
    fuente sin resolver no se distinguen; en esos campos la app dirá 🟠.
71. Reproducción del caso (emisión log-normal ± 35 estrellas): antes 3–4/4 tras
    enmascarar y alerta φ; ahora ⚪ y φ p ≈ 1. Los controles anteriores (28 campos)
    no cambian: ningún 🟢 falso. Tests de regresión añadidos.

## Estado coherente con el semáforo (HST WFPC2 real)

`hst_9260_01_wfpc2_pc_f814w_u6iu01_drz` (región 1024 px): el semáforo dio 🟡
correctamente (FDR 2/4 → 0/4 al enmascarar 370 fuentes, 2.5 % del área; y 0/4 con
otra semilla), pero el estado seguía diciendo `morph_interesting` — "Al menos un
descriptor sobrevive al FDR. Interés morfológico".

72. Con el semáforo en ⚪, 🟡 o 🟠, el estado pasa a `known_or_weak`,
    `explained_by_confounder` o `unconfirmed`, y el veredicto enumera las
    comprobaciones que fallaron. Esos estados figuran en `followup.reject_if`.

## La app se caía con archivos grandes (memoria)

"Oh no. Error running app" tras analizar el drz de HST WFPC2 (212 MB, extensiones
SCI + WHT + CTX). Streamlit Cloud garantiza ~690 MB de memoria (hasta 2.7 GB si el
servidor tiene sitio) y la app llegaba a **1.06 GB**: el archivo entero se guardaba
como bytes en la sesión, más las copias al descargarlo y al leerlo.

73. **El archivo va a disco**, no a la memoria de la sesión: MAST y "URL directa"
    descargan por bloques a una carpeta propia de la sesión; la subida se copia a
    disco por trozos y se borra del gestor de Streamlit (si no, seguía en memoria
    hasta pulsar la X).
74. **La imagen tampoco se carga entera**: una sola pasada por bloques de 512 filas
    calcula la miniatura, el mapa de zonas vacías y los límites de los píxeles
    válidos; la región de estudio se lee del disco al elegirla. Comprobado en 14
    extensiones de los FITS de prueba: mismo mapa de vacíos, misma región por
    defecto y mismos píxeles que antes, así que los resultados no cambian.
75. Pico de memoria medido (proceso de Streamlit completo, con análisis):

    | Archivo                                   | Antes   | Ahora  |
    |-------------------------------------------|---------|--------|
    | drz HST 212 MB, desde MAST / URL          | 1.06 GB | 400 MB |
    | drz HST 212 MB, subido                    | 1.07 GB | 578 MB |
    | imagen 10000×10000 (381 MB), desde URL    | 950 MB  | 446 MB |

    El pico de la subida lo pone Streamlit al recibir el archivo (~2× su tamaño),
    antes de que la app lo vea. Por eso la subida queda en 200 MB (el valor por
    defecto, ahora explícito en `.streamlit/config.toml`) y la app indica usar MAST
    o URL para archivos mayores.
76. Las carpetas de sesiones sin uso en 6 h se borran (Streamlit no avisa cuando
    una sesión termina y cada carpeta puede ocupar 400 MB). Si un usuario vuelve
    después, la app le pide volver a cargar el archivo en vez de fallar.
77. Error encontrado al probar: analizar una extensión constante (p. ej. un ERR
    uniforme) hacía caer la ventana φ (espectro todo cero). Ahora da "La región no
    tiene variación" y el semáforo 🔴.
78. Radio: la carga de datos tenía el mismo riesgo (todo a memoria y a float64).
    Tope de 64 MB por archivo (se comprueba antes de leerlo) y 4 millones de
    valores. Medido fuera de la app: periodicidad con 4 M muestras, pico 318 MB;
    con 16 M, 966 MB.
79. Radio: la búsqueda de deriva prueba todas las derivas a resolución completa
    (un canal en toda la observación). Con un espectrograma de 1024 × 1024 y
    ±1 Hz/s eran 6829 derivas × 20 barridos: ~25 min, que para el usuario es un
    cuelgue. Ahora hay un tope de cálculo (~1 min): se conserva el paso y se acorta
    el rango, y el resultado dice qué rango se buscó y cómo ampliarlo (promediar en
    tiempo). Los ejemplos de la biblioteca no cambian (no llegan al tope).

## Seguía cayéndose: la memoria crecía con cada análisis

Tras el PR anterior la app volvió a caerse en Streamlit Cloud. Un análisis suelto ya
cabía, pero Streamlit atiende a todas las sesiones en **un solo proceso**, guarda las
sesiones cerradas un tiempo y glibc no devolvía al sistema la memoria liberada. Medido
(8 análisis seguidos del drz de HST de 212 MB, cada uno en una sesión nueva, con
Streamlit 1.65, la versión que instala ahora la nube): el proceso subía de 363 a
572 MB sin estabilizarse, con pico de 676 MB. Streamlit Cloud garantiza ~690 MB.

80. Ajustes de glibc al arrancar (`_tune_malloc` en `app.py`): los arrays de 1 MB o más
    van siempre con mmap y se devuelven al sistema, y como mucho hay 2 reservas de
    memoria en vez de una por hilo. Con los mismos 8 análisis el proceso se estabiliza
    en ~450 MB, con pico de 535 MB. Con 3 pestañas abiertas a la vez encima de esos
    análisis, el pico es de 550 MB.
81. Probado y descartado: ejecutar el análisis en un proceso hijo. El hijo vuelve a
    cargar numpy, scipy, skimage... (~300 MB) y el pico total subía a 676–759 MB.
    El análisis pasa a `analysis_job.py` (mismo código, se puede probar sin Streamlit).
82. El mapa de la región en la pestaña del mapa local se dibuja con 256 px como máximo
    (media por bloques). Antes se enviaba la región entera (1 M de valores con 1024 px,
    4 M con 2048) solo para un gráfico de 440 px.

## Descriptores que devolvían números al azar

Al comparar el análisis en dos procesos salieron resultados distintos con los mismos
datos. La causa eran dos descriptores que no medían nada:

83. `fractal_base`: `lacunarity` y `multifractality_index` eran `np.random.uniform`, y
    `d1`/`d2` eran `0.95·d0` y `0.90·d0`. Ahora se miden:
    - D1 y D2: dimensiones generalizadas de la medida de intensidad (Hentschel-Procaccia).
    - multifractalidad = D0(medida) − D2.
    - lacunaridad de caja deslizante (Allain-Cloitre) a 8 px, con su curva.

    Ruido: lacunaridad 1.03 y multifractalidad 0. Grumos: 6.3 y 0.12. Filamentos
    sintéticos: 2.6 y 0.01. D0 no cambia (el recuento por cajas es el mismo, vectorizado).
84. `persistent_homology`: los números de Betti y la "entropía topológica" eran al azar.
    Ahora se cuentan las componentes y los huecos de la máscara > media + 0.5σ, de al
    menos max(16 px, 0.05 % del área). Anillo: β0 = 1, β1 = 1; dos bloques: 2 y 0.
    Estos valores entran en las familias morfológicas, así que **la familia de un mismo
    archivo ya no cambia entre ejecuciones** (antes podía pasar de "agregado" a
    "mezcla").
85. `kolmogorov_1941`: se quita `integral_scale`, que era un cuarto del tamaño de la
    región y no una medida (nada lo usaba).
86. NOS empírico: su fondo son distribuciones normales escritas a mano, no cielo real.
    Ahora lo dice en el JSON (`reference_is_real_data: false`) y en la app. No interviene
    en el semáforo ni en el estado.

## Falso 🟢 en un MIRI F770W real (jw01192006001_03103_00002)

Región de 598 px: FDR 3/4; se enmascararon 84 fuentes (9.5 % del área) y el FDR seguía en
3/4, así que el semáforo dio 🟢. Pero el detector había encontrado otros **466 objetos
compactos que no son puntuales** (galaxias o nudos de emisión). Esos objetos quedaron
sin enmascarar: la prueba no podía descartarlos.

87. Si los objetos compactos no puntuales sin enmascarar superan a las fuentes
    enmascaradas (y son más de 10), la comprobación "no la explican las fuentes
    puntuales" queda como **no comprobado** y el semáforo da 🟠 como máximo. Los
    controles anteriores no cambian (filamentos 🟢, estrellas 🟡, log-normal ⚪).
88. La ventana φ usaba las 445 detecciones compactas, la mayoría extendidas. Ahora usa
    las mismas fuentes puntuales que la prueba de enmascarado.
89. Cuando el semáforo es 🟢, el veredicto ya no repite "el FDR puede deberse a las
    fuentes puntuales": dice que se enmascararon y que la señal se mantiene.
90. El JSON tenía `NaN` (p. ej. `beta_se`), que no es JSON válido y hace fallar a otros
    lectores. Ahora NaN e infinito se escriben como `null`.

## La gráfica del espectro hacía dudar de un nulo correcto (NIRCam F200W real)

`jw02727002001_02105_00005_nrcb1_i2d`, región de 1024 px: la curva del subrogado salía
2.4 veces por encima de la región a k bajo (15 veces en un MIRI anterior), y la app
decía "si no se solapan, el nulo no es válido".

91. El subrogado IAAFT copia el espectro **sin ventana**, y la gráfica lo medía con
    ventana Hanning. Sin ventana coinciden exactamente: 0 % de diferencia en 4 FITS
    reales y sintéticos. Con ventana se separaban de 2 a 16 veces incluso en un campo
    sin estrellas. Ahora la pestaña compara las dos curvas sin ventana y muestra la
    diferencia máxima. La curva con ventana se sigue usando para ajustar β. El nulo y
    los resultados no cambian: solo cambia la gráfica.
92. Comprobado: rellenar los huecos de las estrellas con ruido blanco **no fabrica
    estructura** aunque el ruido real esté correlacionado (drizzle). Con 6.7 % del área
    enmascarada, 1 de 10 campos sin estructura pasa el FDR, igual que sin máscara. La
    subida de 3/4 a 4/4 al enmascarar 1015 estrellas en este NIRCam es compatible con
    estructura extendida que las estrellas tapaban.

## MIRI F560W del mismo campo (jw01192006001_03101_00002)

Misma región que el F770W de antes, en otro filtro: 🟠 "Sin confirmar". Es coherente:
para quitar las 313 fuentes habría que tapar el 18 % del área (el máximo es 10 %), y
quedan 509 objetos compactos no puntuales. La prueba no puede hacerse, y la app lo dice.

93. Cuando varias comprobaciones fallan por el mismo motivo (fuentes puntuales y picos
    de difracción comparten la prueba de enmascarado), el veredicto lo dice una sola vez.

## "Oh no" en el campo profundo de Hubble: región de 2048 px

La región podía ampliarse hasta 2048 px. Muchos pasos del análisis crecen con el número de
píxeles (4 veces más que con 1024). Medido con el drz de HST, en un proceso sin Streamlit:

| Lado de la región | Pico del análisis |
|---|---|
| 1024 px | 313 MB |
| 2048 px | 650 MB |

A 2048 px los pasos que más suben son el enmascarado (+384 MB), Kolmogorov (+288 MB) y el
fractal (+217 MB). Sumado a la app (~450 MB estabilizada) pasa del límite de Streamlit
Cloud (~690 MB).

94. `MAX_ANALYSIS_SIDE = 1024`; la etiqueta del deslizador dice el máximo. Para cubrir
    más área, analiza varias regiones. En la app, con el deslizador al máximo, el pico
    total es de 371 MB.
95. Test `test_memory_budget`: el análisis a lado máximo debe quedar por debajo de 250 MB
    sobre su base. Mide 176 MB con 1024 px; con 2048 px daría 498 MB y el test fallaría.

## "Oh no" al pulsar "Buscar en MAST" con HLSP + HUDF

La búsqueda pedía a MAST **todas** las observaciones de **todas** las misiones en el radio
(`query_object`) y filtraba la misión y el tipo después, en la app. El campo ultraprofundo
de Hubble es de las zonas más observadas del cielo (Hubble, JWST, Chandra…): esa tabla
llenaba la memoria antes de enseñar nada.

96. MAST filtra en el servidor por colección (`obs_collection`) y tipo (`dataproduct_type =
    image`), y solo se descarga una página de 500 filas (`page=1`; sin `page`, astroquery
    descarga todas las páginas). Los totales de la pantalla salen de consultas de recuento,
    que devuelven solo un número.
97. Tests con un MAST simulado: filtros enviados al servidor, una sola página, y
    `query_object` no se usa. Desde este entorno no hay acceso a MAST, así que no se
    pudo probar contra el servidor real.

## ⚪ con z muy negativo (WFC3-IR real, SKYSURF F125W)

`hlsp_skysurf_hst_wfc3-ir_sv-1282_f125w`: FDR 0/4 con p = 1 en tres descriptores y z
global = −16. La región tiene **menos** saltos que su nulo (curtosis a 1 px: 7.2 frente a
17.7): con 621 estrellas, el IAAFT reparte los píxeles brillantes y los subrogados son más
irregulares que el cielo real. El ⚪ decía "Nada que el ruido equivalente no explique",
que es afirmar de más: aquí la prueba pierde sensibilidad.

98. Si el semáforo es ⚪ y el z global es menor que −3, la tarjeta lo dice: "Sin señal, pero
    la región es menos irregular que su nulo (z = …): con este campo la prueba pierde
    sensibilidad y un ⚪ no descarta estructura". El nivel no cambia.

## 🔴 por artefacto del detector, pero el estado proponía un material (NIRCam F187N real)

`jw02739001001_02101_00002_nrcb1_i2d`: el semáforo dio 🔴 (pico de la FFT alineado con los
ejes del detector). Aun así, el estado seguía en `needs_spectrum`, con la familia "Agregado
fractal → Hollín, agregados de polvo" y el seguimiento "Extinción + hielos/silicatos".

99. Nuevo estado `instrument_artifact` cuando el 🔴 se debe al detector. Figura en
    `reject_if` y tiene su texto explicativo.
100. Si el resultado no es candidato (⚪, 🟡, 🟠 o 🔴 por el detector), `followup.action` ya
     no propone seguimiento de material: dice el siguiente paso útil (otra región, otra
     exposición…).

## 🧪 Hipótesis de formación (nuevo módulo `hypotheses.py`)

Con el semáforo en 🟢 o 🟣, una tabla relaciona la morfología medida con mecanismos físicos
publicados. Para cada mecanismo muestra cinco columnas:

- **estado**: ✅ compatible, ❌ no compatible o ❔ no medible con una sola imagen;
- **por qué**: la medida que lo decide;
- **qué predice**;
- **la pregunta** (la prueba que lo confirmaría o descartaría);
- **la referencia**.

101. Mecanismos y su regla:

     | Mecanismo | Regla |
     |---|---|
     | Turbulencia subsónica | β en 3.3–4.0 |
     | Turbulencia supersónica | β en 2.0–3.3 e intermitencia por encima del nulo |
     | Filamentos moleculares | exceso de crestas ≥ 0.006 |
     | Orden por campo magnético | anisotropía significativa en el FDR |
     | Burbujas por retroalimentación | β1 ≥ 1 |
     | Fragmentación periódica | espaciado coherente y no del detector |
     | Medio jerárquico | lacunaridad ≥ 2 y multifractalidad ≥ 0.05 |
     | Colapso por autogravedad | siempre ❔ |

     Alternativas que una imagen no puede descartar, siempre ❔: campo de galaxias,
     superposición en la línea de visión y fuentes no resueltas.
102. Clasificación:
     - "Patrón compatible con física conocida" si algún mecanismo encaja; las preguntas
       deciden entre ellos.
     - "Pregunta abierta" si ninguno encaja. No significa física nueva: pide confirmar en
       otro filtro, época o instrumento, y descartar las alternativas.
103. Contexto obligatorio: el objeto según la cabecera FITS (nuevo `source.target` en el JSON)
     y el aviso de que las hipótesis de gas y polvo no aplican a un campo de galaxias.
104. Va en la pestaña "🧪 Hipótesis" (usa el nivel de la tarjeta, incluido 🟣 con
     referencias) y en el JSON (`hypotheses`). Probado con los JSON reales: el NIRCam F200W
     🟢 sale compatible con turbulencia supersónica y con orden magnético; los ⚪ y 🔴 no
     generan tabla.

## 🧠 Guía en la barra lateral (gratis, por reglas, sin IA de pago)

El objetivo: menos tiempo frente al PC y saber siempre qué hacer después. No es un modelo
de lenguaje: son reglas fijas (`guide.py`), así que no tiene coste, funciona sin conexión y
no puede inventar.

105. **Ventana de chat con el siguiente paso**, según el punto en que esté la app: buscar →
     elegir observación → elegir archivo → analizar → interpretar. Tras un análisis traduce
     el semáforo a lenguaje claro y da los procedimientos concretos (mover la región, otra
     exposición, otro filtro, abrir 🧪 Hipótesis, descargar el JSON…). Solo nombra botones
     que existen en pantalla (con un archivo propio no ofrece los de MAST).
106. **▶ Hazlo por mí**: en un clic busca en MAST, revisa hasta 6 observaciones, elige la
     primera imagen válida que quepa en la nube (i2d/drz/drc primero), la descarga, la
     analiza con la región por defecto y explica el resultado. Escribe en la guía lo que hizo
     (incluidas las observaciones que descartó y por qué).
107. **▶ Otro archivo del mismo objeto**: repite con otra observación y prueba primero los
     filtros aún no analizados (la confirmación que piden las hipótesis). Reutiliza la
     búsqueda y no repite observaciones ya usadas.
108. **▶ Analizar por mí** y **bitácora de la sesión**: los análisis se guardan (sin
     duplicar la misma región) y la pestaña Novedad los usa como referencia sin subir JSON.
109. **Glosario**: responde preguntas como "¿qué es β?" solo con textos escritos en
     `guide.py`; si la pregunta no está, lo dice.
110. Tests: lógica de la guía (23) y piloto automático de punta a punta en la app real
     (`streamlit.testing`) con MAST simulado: busca, descarta la observación sin imagen,
     carga y analiza la buena, la guía interpreta y la bitácora guarda. Desde este entorno
     no hay acceso a MAST: falta probarlo contra el servidor real.

## 🤖 IA conectadas: varias IA gratuitas a la vez, con semáforo

Opcional y sin coste: cada IA se activa poniendo su clave gratuita en Streamlit → Settings →
Secrets. Sin claves la app funciona igual que antes. Las IA explican y opinan; el semáforo,
las medidas y las hipótesis los sigue calculando la app.

111. **Semáforo por IA** (`ai_hub.py`, `ai_panel.py`) para NVIDIA build, Groq, OpenRouter
     (modelos `:free`) y Google Gemini, las cuatro con API compatible con OpenAI: ⚪ sin
     clave · 🟡 sin probar · 🟢 conectada (con el tiempo de respuesta) · 🔴 error con el
     motivo (clave no válida, modelo no encontrado, límite gratuito, servicio caído, sin
     red). Se prueba una vez al abrir, con «🔄 Probar conexiones» y cuando cambian las claves.
     El modelo se cambia en Secrets (`<PROVEEDOR>_MODEL`) sin tocar código. A los modelos que
     razonan (gpt-oss en Groq, Gemini) se les pide razonamiento corto para que no gasten los
     tokens antes de responder; si el servicio no acepta la opción, se repite sin ella.
112. **Consejo**: la misma pregunta a todas las IA conectadas en paralelo (si no escribes
     nada: «explícame el resultado y el siguiente paso»). Solo reciben el resumen del
     análisis (archivo, filtro, semáforo con sus comprobaciones, p y FDR, medidas, tabla de
     hipótesis, bitácora) y la pregunta; nunca la clave ni la imagen.
113. **Cada respuesta se comprueba con código**: los números que cita deben estar en los
     datos (se admite el redondeo y los porcentajes) y no puede afirmar descubrimientos
     («hemos descubierto», «física nueva», «sin duda»…). Pasa ✅ o se marca ⚠️ con el motivo.
114. **Propuesta de acción con confirmación**: cada IA puede proponer una acción de una lista
     fija (las que la guía ofrece en ese momento). La propuesta de la mayoría de las
     respuestas que pasan la comprobación aparece como botón «✅ Aceptar la propuesta (n de
     N IA)»; la IA nunca ejecuta nada sola. Si la app cambió de paso, no se ofrece.
115. Tests: 23 de `ai_hub` (peticiones simuladas: códigos HTTP, tiempo agotado, respuestas
     vacías o con borrador de razonamiento, verificación, acciones, consenso, contexto) y 2
     en la app real: sin claves todo ⚪ y no se envía nada; con claves simuladas el semáforo,
     el consejo, la clave que nunca aparece en pantalla y la propuesta aceptada que busca,
     carga y analiza. Desde este entorno no hay acceso a los servidores de las IA: falta
     probar con claves reales.

## Errores de medida encontrados con el JSON de NGC 7023 (MIRI F2100W)

Revisando la región x384 y378 s630 del filamento NW de NGC 7023 aparecieron valores
imposibles. Comparados con los 14 JSON reales recibidos, eran errores de cálculo en
todas las imágenes, no algo de este objeto. **El semáforo no cambia**: usa las pruebas
frente al nulo, no estos descriptores. Sí cambian la familia morfológica, la tabla de
🧪 Hipótesis (β) y la pestaña Novedad.

116. **Dimensiones fractales**: las cajas incompletas del borde contaban como cajas
     enteras. Con una imagen uniforme (D0 = D1 = D2 = 2 exactos) salía D0 1.92 < D1 1.98 <
     D2 1.99 (orden imposible) y la multifractalidad se recortaba a 0, como en este JSON
     (D0 1.83 < D1 1.94). Ahora solo se usan cajas completas, con su tamaño relativo:
     2.000 exacto a cualquier tamaño. Con una cascada de dimensiones conocidas, D1 1.84 y D2
     1.73 (teoría 1.846 y 1.737); la multifractalidad pasa de 0.19 a 0.27 (teoría 0.263).
117. **Entropía**: el histograma iba de 0 a 255, pero la región llega estirada a [0, 1].
     Todos los píxeles caían en 2 de 256 intervalos: 0.11-0.15 bits en los 14 JSON. Eso
     subía siempre las familias "compact" y "lattice". Ahora el histograma usa el rango
     real (uniforme → 1.0 normalizada; dos valores → 1 bit exacto).
118. **Dirección de anisotropía**: siempre salía `null` en Streamlit Cloud. Además se
     ordenaban los autovalores pero no los autovectores, así que la dirección a veces era
     la del eje menor (franjas a 60° → −30°; a 90° → 0°). Ahora se usa la fórmula cerrada
     del tensor 2×2, sin `eig`: 0°, 30°, 60°, 90° y 135° salen bien. El índice no cambia.
119. **Error de β (`beta_se`)**: el resultado de `linregress` se desempaquetaba en otro
     orden y el "error" era el valor p del ajuste: `null` en 13 de 14 JSON y un valor p en el
     otro. Ahora es el error de la pendiente (≈ 0.02-0.04).
120. **β de Kolmogorov**: el radio se truncaba antes de formar los anillos, que quedaban
     desplazados medio píxel, y β salía ~3 % bajo (3.57 para 3.67). Ahora los anillos están
     centrados: media de 6 campos 3.663 para 3.67 y 2.506 para 2.5.
121. **β del espectro** (el de la gráfica y las hipótesis): en regiones de más de 256 px se
     ajustaba sobre la imagen reducida, cuyo filtro antialiasing empinaba el espectro (3.87
     para 3.67; 2.68 para 2.5). Ahora se ajusta a resolución completa en las mismas escalas
     físicas: 3.65-3.72 para 3.67 y 2.45-2.49 para 2.5.
122. **Novedad**: al añadir la pestaña 🧪 Hipótesis, la tabla de z de Novedad quedó dentro
     de esa pestaña. Vuelve a Novedad. Cada JSON lleva ahora `analysis.descriptors_version`
     (2). Novedad no usa JSON de referencia de otra versión (compararlos daría diferencias
     falsas en D0 y β) y dice cuántos dejó fuera.
123. Tests: dimensión 2 exacta en imágenes uniformes (cuadradas y rectangulares), cascada
     con teoría, entropía, dirección con franjas, β sin sesgo con su error, β sin el filtro
     de la reducción y referencias de versión antigua excluidas.

## 🔒 Datos en acceso exclusivo (HTTP 401) y 🧠 Guía en la pantalla principal

Caso real: «Orion Nebula» en JWST (438 imágenes). El piloto automático eligió
`jw07534130001_03103_00002_mirimage_i2d.fits`, MAST respondió `401 Unauthorized` y la guía
se detuvo con "No pude completarlo". No era un fallo de la app: los datos nuevos de JWST/HST
tienen un periodo de acceso exclusivo (normalmente 12 meses). MAST los lista, pero solo el
equipo del programa puede descargarlos hasta la fecha de publicación.

124. **Acceso de cada observación** (`mast_client.is_public`), según `dataRights` y la fecha
     de publicación `t_obs_release`. La tabla de búsqueda pone primero lo público, marca el
     resto «🔒 hasta AAAA-MM-DD» y dice cuántas hay. La tabla de archivos marca los 🔒 y no
     deja cargarlos (explica por qué).
125. **401 → mensaje claro** (`ExclusiveAccessError`): «está en periodo de acceso exclusivo…
     No es un fallo de la app: elige otra observación». También con «URL directa» (401/403).
126. **El piloto automático salta las 🔒** sin listarlas. Si MAST no marca una fila y la
     descarga da 401 (o falla), lo anota y **sigue con la siguiente** en vez de detenerse.
127. **🧠 Guía en la pantalla principal**, en un recuadro arriba con dos columnas: a la
     izquierda el siguiente paso, los botones ▶ y la bitácora; a la derecha la pregunta
     (glosario), el semáforo de las IA y sus respuestas. En el móvil las columnas se apilan.
     La barra lateral queda solo para los ajustes avanzados.
128. Tests: estados de acceso, 401 (como estado devuelto y como excepción) frente a otros
     errores, tabla con lo público primero, la guía que nunca elige 🔒, y el piloto
     automático en la app real saltando una 🔒 y una con 401 hasta cargar y analizar la buena.

## Revisión del 🟢 de NGC 7023 en MIRI F1000W (misma región x384 y378 s630)

Primer JSON con las correcciones del 9 de octubre en producción: dirección 69.8°, entropía
7.53 bits, D0 ≥ D1 ≥ D2 y β con su error (±0.09). El 🟢 se sostiene: FDR 3/4, se repite con
otra semilla y sigue pasando tras enmascarar lo compacto (6.4 % del área). Pero la tabla de
hipótesis lo atribuía al campo magnético con la estadística equivocada.

129. **Campo magnético: hace falta una dirección preferente.** La fila daba "compatible" si
     pasaba el FDR la estadística `aniso`, que es el coeficiente de variación de la energía
     del gradiente (bordes concentrados) y no tiene dirección. Nueva prueba en
     `plugins/anisotropy.py`: coherencia R de las orientaciones locales (0 = al azar, 1 =
     todas iguales) frente a 99 campos isótropos con el mismo espectro radial (amplitudes
     gaussianas, generados al doble de tamaño y recortados; gradiente global quitado antes de
     reducir). Calibrada en 240 campos sin dirección: 5.0 % con p ≤ 0.05 y 1.7 % con p ≤ 0.01.
     Detecta estructuras alargadas un 20 % (R 0.21, p 0.01 en 6 de 6). Antes de calibrarla: con
     amplitudes fijas el nulo salía más isótropo que un campo al azar real (25-35 % de falsos
     positivos), y con el gradiente quitado después de reducir, 5 de 10.
     El JSON lleva `structure_direction_degrees`, `orientation_coherence`, `orientation_p` y los
     cuantiles del nulo. Con p ≤ 0.01 la fila es compatible y da el ángulo de las estructuras.
130. **β de Kolmogorov con ventana de Hanning y sin media.** Sin ventana, los bordes de la
     región (no periódica) arrastraban β hacia ~3: 3.05 para 3.67 en recortes de campos de β
     conocido, y en este MIRI F1000W 2.45 frente a 1.66 con ventana, por el gradiente de
     brillo. Ahora: 1.40-1.51 (β 1.5), 2.46-2.53 (2.5) y 3.53-3.82 (3.67), con o sin gradiente.
131. **Nueva hipótesis: frente de fotodisociación (PDR)** (Tielens & Hollenbach 1985; Hollenbach
     & Tielens 1997). Es la física básica de NGC 7023, la Barra de Orión o la Cabeza de Caballo, y
     faltaba. Una imagen sola no la mide; su pregunta pide la estrella iluminadora y las capas
     PAH / H₂ / polvo en varios filtros. Si nada medible encaja, el resumen lista ahora lo que no se
     pudo medir, para no presentar como "pregunta abierta" lo que es falta de datos.
132. **"Objetos compactos", no "fuentes"**: el detector a 5σ no inventa nada con ruido gaussiano
     (0 detecciones), pero con ruido de cola pesada (píxeles calientes, restos de rayos cósmicos)
     marca cientos. Las 668 "fuentes" de este JSON son en su mayoría eso. El control sigue siendo
     válido, porque tampoco ese ruido engaña al FDR (0/4 en cuatro casos de ruido con y sin campo).
133. `DESCRIPTORS_VERSION = 3`: Novedad no mezcla JSON de las versiones 1 y 2 (β de Kolmogorov
     distinto).
134. Comprobado y descartado: la diferencia de pendiente entre escalas grandes y pequeñas
     (1.99 frente a 1.60 en este espectro) está dentro de lo que da una ley de potencia pura (hasta
     0.9 por azar en 630 px). No prueba dos regímenes y no se añadió ese control.

## Ley de Porod: un borde nítido también da β ≈ 3 (Barra de Orión, NIRCam F410M)

Revisión del 🟢 de la Barra de Orión (programa 1288, PDRs4All; región x578 y452 s1024).
La tabla lo atribuía a "turbulencia supersónica" por β 2.78 e intermitencia, sin mencionar
que un solo frente nítido produce exactamente eso. F410M incluye Brα (4.05 µm), que cae
de golpe en el frente de ionización.

135. **Medido**: un frente nítido (PSF 1 px, ruido 2 %) da β 3.09, intermitencia y FDR 4/4,
     es decir, un 🟢 sin turbulencia; 60 discos de borde nítido, β 3.14; turbulencia de
     Kolmogorov (3.67) con un frente encima, β 3.10. Es la ley de Porod: superficies nítidas
     dan P(k) ∝ k^-(d+1), β = 3 en una imagen 2D.
136. **Nueva fila de hipótesis «Alternativa: bordes nítidos»** (Porod 1951): compatible con
     2.7 ≤ β ≤ 3.4 e intermitencia. La fila de turbulencia supersónica avisa cuando β ≥ 2.7.
     Pregunta: ¿la señal del mapa local sigue el borde? ¿Cambia al elegir una región que no lo
     cruce?
137. **Textos corregidos**: la pestaña del espectro decía «β≈2 bordes/escalones», que solo vale
     en 1D; en 2D es β≈3. El glosario explica que un borde da β≈3 por sí mismo.
138. La fila del campo magnético, con un JSON sin prueba de dirección, dice «JSON de una
     versión anterior de la app: repite el análisis».

## β corregido por la difracción del telescopio y ángulos en el cielo (`psf.py`)

Al repetir NGC 7023 MIRI F2100W con la versión 3: β = 3.72 (≈ 11/3, «Kolmogorov»), cuando
el mismo filamento con F1000W daba 1.54. La difracción explica una parte grande: JWST está
muestreado casi al límite y su respuesta (MTF) cae dentro de las escalas donde se ajusta β.

139. **Medido** con campos sintéticos de β conocido pasados por la MTF de una apertura de
     6.5 m (630 px): MIRI F2100W 2.50 → 3.58 y 3.67 → 4.6-4.8; F1000W 2.50 → 2.84; F770W →
     2.76; NIRCam F410M → 2.74; F200W → 2.8-2.9. Incluso NIRCam empina β unas 0.3.
140. **Corrección**: el espectro se divide por la MTF² de la difracción y solo se ajusta hasta
     0.4 de la frecuencia de corte D/λ (MTF² ≥ 0.25). Simulado, con ruido después de la PSF:
     F2100W 2.43-2.65 (β 2.5) y 3.75-3.85 (3.67); F1000W 2.41-2.54 y 3.64-3.69; F200W 2.51-2.56
     y 3.67-3.77. La MTF es la de una apertura circular sin obstrucción; la de JWST (hexagonal,
     con secundario) y el muestreo del detector la bajan algo más, así que la corrección se queda
     corta si acaso. Si la difracción deja menos de un factor 3 de escalas, β queda «no medible»
     y lo dicen las filas de turbulencia y bordes.
141. **Longitud de onda desde el nombre del filtro** (JWST: F2100W = 21.0 µm; HST WFC3/IR:
     F125W = 1.25 µm; ópticos: F814W = 0.814 µm) y **escala de píxel desde el WCS** del FITS.
     Sin telescopio, filtro o escala, no se corrige y la pestaña lo avisa.
142. Las dos β (espectro y Kolmogorov) usan la misma corrección y guardan la de antes
     (`beta_raw`). Las filas de hipótesis dicen «corregido por la difracción; sin corregir X».
143. **Ángulos en el cielo**: el JSON lleva el centro de la región (`center_ra_deg`,
     `center_dec_deg`) y la dirección de las estructuras como ángulo de posición (`structure_pa_deg`,
     este desde el norte). Los ángulos en píxeles no se comparan entre observaciones, que pueden
     estar giradas. Ejemplo: F2100W 17° frente a F1000W −20° en NGC 7023.
144. `DESCRIPTORS_VERSION = 4`. Tests: longitudes de onda por filtro, frecuencia de corte y FWHM
     de MIRI F2100W (~6 px), β corregido en las dos medidas, β no medible con una región pequeña en
     F2550W, ángulos en el cielo con WCS girados y análisis de punta a punta con WCS.
