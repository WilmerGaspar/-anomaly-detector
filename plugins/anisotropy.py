import numpy as np
from scipy import ndimage

def calculate_anisotropy(image):
    """
    Detecta direcciones preferenciales en la estructura
    """
    # Asegurar que es array 2D
    if len(image.shape) > 2:
        image = np.mean(image, axis=2)
    
    # Calcular gradientes
    gy, gx = np.gradient(image.astype(float))
    
    # Suavizar para reducir ruido
    sigma = 2.0
    Ixx = ndimage.gaussian_filter(gx**2, sigma=sigma)
    Ixy = ndimage.gaussian_filter(gx*gy, sigma=sigma)
    Iyy = ndimage.gaussian_filter(gy**2, sigma=sigma)
    
    # Anisotropia local del tensor de estructura 2x2 [[Ixx, Ixy], [Ixy, Iyy]] con la formula
    # cerrada (exacta para matrices simetricas 2x2):
    #   (l1 - l2) / (l1 + l2) = sqrt((Ixx - Iyy)^2 + 4 Ixy^2) / (Ixx + Iyy)
    #   angulo del autovector mayor = 0.5 * atan2(2 Ixy, Ixx - Iyy)
    # Antes se usaba np.linalg.eig punto a punto: ordenaba los autovalores pero no los
    # autovectores (la direccion salia a veces del eje menor) y, en Streamlit Cloud, la
    # direccion dominante salia vacia (null) en los 14 JSON reales recibidos.
    step = max(1, min(image.shape) // 50)  # Muestreo adaptativo (mismos puntos que antes)
    sxx, sxy, syy = Ixx[::step, ::step], Ixy[::step, ::step], Iyy[::step, ::step]
    trace = sxx + syy
    ok = np.isfinite(sxx) & np.isfinite(sxy) & np.isfinite(syy) & (trace > 1e-10)
    anisotropies = (np.sqrt((sxx - syy) ** 2 + 4.0 * sxy ** 2)[ok] / trace[ok]).clip(0.0, 1.0)
    directions = 0.5 * np.arctan2(2.0 * sxy[ok], (sxx - syy)[ok])

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
    
    return {
        "anisotropy_index": float(mean_anisotropy),
        "dominant_direction_degrees": float(np.degrees(dominant_angle)) if dominant_angle is not None else None,
        "isotropy_score": float(1 - mean_anisotropy),
        "direction_variance": float(direction_variance) if direction_variance is not None else None,
        "interpretation": interp,
        "complexity_score": float(np.std(anisotropies))
    }
