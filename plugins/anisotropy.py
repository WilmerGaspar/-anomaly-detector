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
    
    # Calcular anisotropía local
    anisotropies = []
    directions = []
    
    step = max(1, min(image.shape) // 50)  # Muestreo adaptativo
    
    for i in range(0, image.shape[0], step):
        for j in range(0, image.shape[1], step):
            M = np.array([[Ixx[i,j], Ixy[i,j]], 
                         [Ixy[i,j], Iyy[i,j]]])
            
            if not np.isnan(M).any() and not np.isinf(M).any():
                try:
                    w, v = np.linalg.eig(M)
                    w = np.sort(w)[::-1]  # Mayor primero
                    
                    if w[0] + w[1] > 1e-10:
                        aniso = (w[0] - w[1]) / (w[0] + w[1])
                        anisotropies.append(aniso)
                        
                        # Dirección del autovector mayor
                        angle = np.arctan2(v[1,0], v[0,0])
                        directions.append(angle)
                except Exception:
                    continue
    
    if len(anisotropies) == 0:
        return {
            "anisotropy_index": 0.0,
            "dominant_direction_degrees": None,
            "isotropy_score": 1.0,
            "direction_variance": None,
            "interpretation": "No calculable - imagen uniforme",
            "complexity_score": 0.0
        }
    
    anisotropies = np.array(anisotropies)
    directions = np.array(directions)
    
    # Índice global de anisotropía
    mean_anisotropy = np.mean(anisotropies)
    
    # Dirección dominante (circular, usar estadística circular)
    if len(directions) > 0:
        sin_mean = np.mean(np.sin(2*directions))
        cos_mean = np.mean(np.cos(2*directions))
        dominant_angle = 0.5 * np.arctan2(sin_mean, cos_mean)
        # corregir cálculo de varianza circular
        direction_variance = 1 - np.sqrt(sin_mean**2 + cos_mean**2)
    else:
        dominant_angle = None
        direction_variance = None
    
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
