import numpy as np
from scipy.stats import entropy as scipy_entropy

def calculate_entropy(image, bins=256):
    """
    Calcula entropía de Shannon y métricas de información
    """
    # Convertir a 1D y normalizar
    if len(image.shape) > 2:
        image = np.mean(image, axis=2)
    
    data = image.flatten().astype(float)
    
    # Histograma sobre el rango real de los datos. Antes era siempre 0-255, pero la app
    # pasa la region estirada a [0, 1]: todos los pixeles caian en los 2 primeros de 256
    # intervalos y la entropia salia ~0.14 bits en cualquier imagen (los 14 JSON reales
    # recibidos: 0.108-0.145), que ademas subia las familias "compact" y "lattice".
    finite = data[np.isfinite(data)]
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 0.0)
    if hi > lo:
        hist, bin_edges = np.histogram(finite, bins=bins, range=(lo, hi))
        hist = hist[hist > 0]  # Eliminar bins vacíos
        # Entropía de Shannon (bits)
        shannon_entropy = float(scipy_entropy(hist, base=2))
    else:
        shannon_entropy = 0.0                     # imagen constante: un solo valor
    
    # Entropía máxima (uniforme)
    max_entropy = np.log2(bins)
    
    # Normalizar
    normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0
    
    # Entropía conjunta (correlación espacial simple)
    # Diferencias entre píxeles vecinos
    h, w = image.shape[0], image.shape[1]
    if w > 1:
        diff_h = np.diff(data.reshape(h, w), axis=1).flatten()
    else:
        diff_h = np.array([])
    if h > 1:
        diff_v = np.diff(data.reshape(h, w), axis=0).flatten()
    else:
        diff_v = np.array([])
    differences_valid = (diff_h.size + diff_v.size > 0)
    differences = np.concatenate([diff_h, diff_v]) if differences_valid else np.array([])
    
    if differences.size > 0:
        hist_diff, _ = np.histogram(differences, bins=bins, density=True)
        hist_diff = hist_diff[hist_diff > 0]
        joint_entropy = scipy_entropy(hist_diff, base=2)
    else:
        joint_entropy = 0.0
    
    # Información mutua aproximada
    mutual_info = shannon_entropy - (joint_entropy / 2) if joint_entropy > 0 else 0
    
    # Nivel de complejidad
    if normalized_entropy > 0.9:
        level = "Muy alta - distribución casi uniforme, máxima incertidumbre"
    elif normalized_entropy > 0.7:
        level = "Alta - distribución diversa, estructura compleja"
    elif normalized_entropy > 0.5:
        level = "Media - distribución balanceada"
    elif normalized_entropy > 0.3:
        level = "Baja - distribución concentrada, estructura simple"
    else:
        level = "Muy baja - distribución pico, alta predictibilidad"
    
    # Interpretación física
    if normalized_entropy > 0.8 and mutual_info > 2:
        interpretation = "Sistema desordenado con correlaciones espaciales complejas (turbulencia, caos)"
    elif normalized_entropy > 0.8:
        interpretation = "Sistema desordenado sin correlaciones (ruido térmico)"
    elif normalized_entropy < 0.3:
        interpretation = "Sistema altamente ordenado (estructura cristalina, objeto compacto)"
    elif mutual_info > 3:
        interpretation = "Estructura ordenada con alta complejidad espacial (fractales, filamentos)"
    else:
        interpretation = "Estructura moderada con organización espacial presente"
    
    information_density = float(shannon_entropy / (h * w) * 1000) if h*w>0 else 0.0
    
    return {
        "shannon_entropy_bits": float(shannon_entropy),
        "normalized_entropy": float(normalized_entropy),
        "max_possible_entropy": float(max_entropy),
        "joint_entropy_approx": float(joint_entropy),
        "mutual_information_approx": float(mutual_info),
        "complexity_level": level,
        "interpretation": interpretation,
        "information_density": information_density
    }
