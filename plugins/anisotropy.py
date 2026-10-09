import numpy as np
from scipy import ndimage

def _local_orientation(image, sigma=2.0):
    """Anisotropia local (0-1) y angulo del gradiente (rad) en una rejilla de ~50 x 50 puntos,
    del tensor de estructura 2x2 [[Ixx, Ixy], [Ixy, Iyy]] con la formula cerrada (exacta):
      (l1 - l2) / (l1 + l2) = sqrt((Ixx - Iyy)^2 + 4 Ixy^2) / (Ixx + Iyy)
      angulo del autovector mayor = 0.5 * atan2(2 Ixy, Ixx - Iyy)
    Antes se usaba np.linalg.eig punto a punto: ordenaba los autovalores pero no los
    autovectores (la direccion salia a veces del eje menor) y, en Streamlit Cloud, la
    direccion dominante salia vacia (null) en los 14 JSON reales recibidos."""
    gy, gx = np.gradient(np.asarray(image, dtype=float))
    Ixx = ndimage.gaussian_filter(gx**2, sigma=sigma)
    Ixy = ndimage.gaussian_filter(gx*gy, sigma=sigma)
    Iyy = ndimage.gaussian_filter(gy**2, sigma=sigma)
    step = max(1, min(image.shape) // 50)  # Muestreo adaptativo (mismos puntos que antes)
    sxx, sxy, syy = Ixx[::step, ::step], Ixy[::step, ::step], Iyy[::step, ::step]
    trace = sxx + syy
    ok = np.isfinite(sxx) & np.isfinite(sxy) & np.isfinite(syy) & (trace > 1e-10)
    aniso = (np.sqrt((sxx - syy) ** 2 + 4.0 * sxy ** 2)[ok] / trace[ok]).clip(0.0, 1.0)
    return aniso, 0.5 * np.arctan2(2.0 * sxy[ok], (sxx - syy)[ok])


def _coherence(directions):
    """Longitud del vector medio de los angulos dobles: 0 = orientaciones al azar, 1 = todas iguales."""
    return float(np.hypot(np.mean(np.sin(2 * directions)), np.mean(np.cos(2 * directions)))) if directions.size else 0.0


def _detrend(img):
    """Quita un plano (gradiente global de brillo) para que no cuente como orientacion. En una
    rejilla completa las coordenadas centradas son ortogonales: ajuste exacto sin matrices grandes."""
    img = np.asarray(img, dtype=float)
    h, w = img.shape
    yc = np.arange(h) - (h - 1) / 2.0
    xc = np.arange(w) - (w - 1) / 2.0
    b = float(yc @ img.sum(axis=1)) / (w * float(yc @ yc)) if h > 1 else 0.0
    c = float(img.sum(axis=0) @ xc) / (h * float(xc @ xc)) if w > 1 else 0.0
    return img - img.mean() - b * yc[:, None] - c * xc[None, :]


def orientation_test(image, n_null=99, seed=0, max_side=256):
    """¿Hay una direccion preferente? Coherencia R de las orientaciones locales frente a campos
    ISOTROPOS con el mismo espectro radial (fase al azar, amplitud promediada en anillos): el
    nulo tiene la misma mezcla de escalas pero ninguna direccion. p = (1 + #(R_nulo >= R)) / (n + 1).
    Es lo que pregunta la hipotesis del campo magnetico. El estadistico "aniso" del FDR no sirve
    para eso: es el coeficiente de variacion de la energia del gradiente (bordes concentrados),
    sin direccion."""
    from scoring import downsample_for_null
    # El plano se quita ANTES de reducir: la reduccion (filtro con reflexion en los bordes) curva
    # un gradiente cerca del borde y despues ya no es un plano (medido: 5 de 10 falsos positivos).
    small = _detrend(downsample_for_null(_detrend(image), max_side=max_side))
    _, d = _local_orientation(small)
    r_obs = _coherence(d)
    h, w = small.shape
    # Espectro radial (potencia media por anillo) en una rejilla 2x mayor: los campos nulo se
    # generan grandes y se recortan, como la region (que es un recorte de un cielo mas grande).
    # Amplitudes gaussianas (ruido blanco filtrado), no fijas: con amplitud fija los pocos
    # modos grandes salian iguales en todas las direcciones y el nulo era MAS isotropo que un
    # campo al azar real (medido: 25-35 % de falsos positivos con beta 3.67).
    power = np.abs(np.fft.fft2(small - small.mean())) ** 2
    kr = np.hypot(np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None])
    nb = max(h, w) // 2
    ring = np.minimum((kr / 0.5 * nb).astype(int), nb)
    cnt = np.bincount(ring.ravel(), minlength=nb + 1)
    pw = np.bincount(ring.ravel(), power.ravel(), minlength=nb + 1) / np.maximum(cnt, 1)
    kc = (np.arange(nb + 1) + 0.5) * 0.5 / nb
    good = (cnt > 0) & (pw > 0)
    good[0] = False
    H, W = 2 * h, 2 * w
    KR = np.hypot(np.fft.fftfreq(H)[:, None], np.fft.fftfreq(W)[None])
    # Interpolacion log-log del espectro medido; por debajo del k minimo, la pendiente de los
    # primeros anillos (escalas mayores que la region).
    lk, lp = np.log(kc[good]), np.log(pw[good])
    slope0 = np.polyfit(lk[:4], lp[:4], 1)[0] if lk.size >= 4 else 0.0
    lK = np.log(np.maximum(KR, 1e-9))
    logP = np.where(lK < lk[0], lp[0] + slope0 * (lK - lk[0]), np.interp(lK, lk, lp))
    amp = np.sqrt(np.exp(logP))
    amp[0, 0] = 0.0
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(int(n_null)):
        big = np.fft.ifft2(np.fft.fft2(rng.standard_normal((H, W))) * amp).real
        y, x = rng.integers(0, H - h + 1), rng.integers(0, W - w + 1)
        null.append(_coherence(_local_orientation(_detrend(big[y:y + h, x:x + w]))[1]))
    null = np.array(null)
    return {"orientation_coherence": r_obs, "orientation_null_mean": float(null.mean()),
            "orientation_null_q95": float(np.quantile(null, 0.95)),
            "orientation_p": float((1 + np.sum(null >= r_obs)) / (len(null) + 1)), "orientation_n_null": int(n_null)}


def calculate_anisotropy(image, with_orientation_test=True):
    """
    Detecta direcciones preferenciales en la estructura
    """
    # Asegurar que es array 2D
    if len(image.shape) > 2:
        image = np.mean(image, axis=2)
    
    anisotropies, directions = _local_orientation(image)

    if anisotropies.size == 0:
        return {
            "anisotropy_index": 0.0,
            "dominant_direction_degrees": None,
            "isotropy_score": 1.0,
            "direction_variance": None,
            "interpretation": "No calculable - imagen uniforme",
            "complexity_score": 0.0
        }
    
    # Índice global de anisotropía
    mean_anisotropy = np.mean(anisotropies)
    
    # Dirección dominante del gradiente (perpendicular a las estructuras alargadas), en
    # grados desde el eje x de la imagen; estadística circular de ejes (ángulo doble).
    sin_mean = np.mean(np.sin(2*directions))
    cos_mean = np.mean(np.cos(2*directions))
    dominant_angle = 0.5 * np.arctan2(sin_mean, cos_mean)
    direction_variance = 1 - np.sqrt(sin_mean**2 + cos_mean**2)
    
    # Interpretación
    if mean_anisotropy < 0.1:
        interp = "Isotrópico - sin dirección preferencial"
    elif mean_anisotropy < 0.3:
        interp = "Débilmente anisotrópico - tendencia direccional suave"
    elif mean_anisotropy < 0.6:
        interp = "Moderadamente anisotrópico - estructura alargada presente"
    else:
        interp = "Fuertemente anisotrópico - dirección dominante clara"
    
    out = {
        "anisotropy_index": float(mean_anisotropy),
        "dominant_direction_degrees": float(np.degrees(dominant_angle)) if dominant_angle is not None else None,
        # Las estructuras alargadas van perpendiculares al gradiente.
        "structure_direction_degrees": float((np.degrees(dominant_angle) + 90.0 + 90.0) % 180.0 - 90.0),
        "isotropy_score": float(1 - mean_anisotropy),
        "direction_variance": float(direction_variance) if direction_variance is not None else None,
        "interpretation": interp,
        "complexity_score": float(np.std(anisotropies))
    }
    if with_orientation_test:
        out.update(orientation_test(image))
    return out
