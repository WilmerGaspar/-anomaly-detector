"""
Plugin FractalBase adaptado para web.
"""
import numpy as np
from scipy import ndimage
from typing import Dict

def safe_log(x, default=0.0):
    if x <= 0 or np.isnan(x) or np.isinf(x):
        return default
    result = np.log(x)
    if np.isinf(result) or np.isnan(result):
        return default
    return result

def safe_divide(a, b, default=0.0):
    if b == 0 or np.isclose(b, 0):
        return default
    result = a / b
    if np.isinf(result) or np.isnan(result):
        return default
    return result

class FractalBase:
    def __init__(self, min_box_size=2, max_box_size=None, num_scales=20):
        self.min_box_size = min_box_size
        self.max_box_size = max_box_size
        self.num_scales = num_scales
    
    def analyze(self, image):
        image = np.asarray(image, dtype=float)
        h, w = image.shape
        max_size = min(h, w)
        
        if self.max_box_size is None:
            self.max_box_size = max_size // 4
        
        scales = np.unique(np.logspace(
            np.log10(self.min_box_size),
            np.log10(self.max_box_size),
            self.num_scales
        ).astype(int))
        
        counts = []
        valid_scales = []
        grids = set()
        
        for scale in scales:
            # Al menos 2 cajas completas por lado, y una sola escala por rejilla (150 y 157 px
            # dan las mismas 4 x 4 cajas en 630 px: el punto contaria dos veces en el ajuste).
            grid = (h // scale, w // scale)
            if scale < 2 or min(grid) < 2 or grid in grids:
                continue
            grids.add(grid)
            
            count = self._box_count(image, scale)
            if count > 0:
                counts.append(count)
                valid_scales.append(scale)
        
        if len(valid_scales) < 3:
            return self._empty_results()
        
        scales = np.array(valid_scales)
        counts = np.array(counts, dtype=float)
        eps = np.array([self._box_eps(image.shape, s) for s in scales])
        
        d0 = self._calculate_dimension(eps, counts)
        
        log_scales = [safe_log(float(s)) for s in eps]
        log_counts = [safe_log(float(c)) for c in counts]
        # Antes d1 = 0.95*d0, d2 = 0.90*d0 y multifractalidad y lacunaridad eran numeros al
        # azar (np.random.uniform): cambiaban en cada ejecucion y movian la familia morfologica.
        # Ahora se miden (ver _generalized_dimensions y _lacunarity).
        gd = self._generalized_dimensions(image, scales)
        lac, lac_curve = self._lacunarity(image)
        
        return {
            'd0': float(np.clip(d0, 0.0, 3.0)),
            'd1': gd['d1'],
            'd2': gd['d2'],
            'dimension_box': float(d0),
            'multifractality_index': gd['delta'],
            'lacunarity': lac,
            'lacunarity_curve': lac_curve,
            'complexity_score': float(np.clip(d0 / 3.0, 0.0, 1.0)),
            'log_scales': log_scales,
            'log_counts': log_counts
        }
    
    # Solo cajas COMPLETAS. Antes el borde sobrante se rellenaba con ceros y esas cajas a
    # medias contaban como cajas enteras: con cajas grandes habia de mas (630 px y cajas de
    # 157: 25 en vez de 16) y todas las dimensiones salian sesgadas. Medido con una imagen
    # uniforme (D0 = D1 = D2 = 2 exactos): D0 1.92 < D1 1.98 < D2 1.99, orden invertido, y
    # la multifractalidad (D0 - D2) se recortaba a 0. El sesgo cambiaba con el lado de la region.
    @staticmethod
    def _box_sums(a, box):
        """Suma por cajas box x box completas; el borde que no llena una caja se descarta."""
        h, w = a.shape
        nh, nw = h // box, w // box
        return a[:nh * box, :nw * box].reshape(nh, box, nw, box).sum(axis=(1, 3))
    
    @staticmethod
    def _box_eps(shape, box):
        """Tamaño de la caja relativo a la zona que cubren las cajas completas: 1/sqrt(nh*nw).
        Con el tamaño en pixeles, el recorte distinto en cada escala volvia a sesgar el ajuste."""
        return 1.0 / np.sqrt((shape[0] // box) * (shape[1] // box))
    
    def _box_count(self, image, box_size):
        """Cajas completas con algun pixel > 0.1."""
        return int(np.count_nonzero(self._box_sums((image > 0.1).astype(np.int32), box_size)))
    
    def _generalized_dimensions(self, image, scales):
        """Dimensiones generalizadas de la medida de intensidad (Hentschel-Procaccia):
        p_i = brillo de la caja / brillo total; D1 = pendiente de sum p log p frente a log e,
        D2 = pendiente de log sum p^2 frente a log e. 'delta' = D0(medida) - D2: 0 para una
        medida uniforme (monofractal), mayor cuanto mas concentrada en pocas zonas."""
        m = np.clip(np.nan_to_num(image, nan=0.0), 0.0, None)
        total = m.sum()
        if total <= 0:
            return {'d1': 0.0, 'd2': 0.0, 'delta': 0.0}
        le, s0, s1, s2 = [], [], [], []
        for e in scales:
            p = self._box_sums(m, int(e)).ravel()
            if p.sum() <= 0:
                continue
            p = p / p.sum()                      # medida de la zona cubierta por cajas completas
            p = p[p > 0]
            le.append(np.log(self._box_eps(m.shape, int(e))))
            s0.append(np.log(len(p)))
            s1.append(float(np.sum(p * np.log(p))))
            s2.append(np.log(np.sum(p ** 2)))
        if len(le) < 3:
            return {'d1': 0.0, 'd2': 0.0, 'delta': 0.0}
        le = np.array(le)
        slope = lambda y: float(np.polyfit(le, np.array(y), 1)[0])
        d0m, d1, d2 = -slope(s0), slope(s1), slope(s2)
        return {'d1': float(np.clip(d1, 0.0, 3.0)), 'd2': float(np.clip(d2, 0.0, 3.0)),
                'delta': float(max(0.0, d0m - d2))}
    
    @staticmethod
    def _lacunarity(image, radii=(2, 4, 8, 16, 32), r_report=8):
        """Lacunaridad de caja deslizante (Allain y Cloitre 1991) de la mascara de estructura
        (pixeles > media + 0.5 sigma): L(r) = <M^2> / <M>^2, M = pixeles de estructura en una
        caja r x r en todas las posiciones. 1 = estructura repartida sin huecos; mayor = grumos
        separados por huecos. Se devuelve L(8 px) y la curva."""
        a = np.nan_to_num(np.asarray(image, dtype=float), nan=0.0)
        mask = (a > a.mean() + 0.5 * a.std()).astype(np.float64)
        if mask.sum() == 0:
            return 0.0, {}
        ii = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1))
        ii[1:, 1:] = mask.cumsum(0).cumsum(1)
        curve = {}
        for r in radii:
            if r >= min(mask.shape):
                continue
            M = ii[r:, r:] - ii[:-r, r:] - ii[r:, :-r] + ii[:-r, :-r]
            mu = M.mean()
            if mu > 0:
                curve[int(r)] = float((M ** 2).mean() / mu ** 2)
        return float(curve.get(r_report, 0.0)), curve
    
    def _calculate_dimension(self, scales, counts):
        log_scales = np.log(scales)
        log_counts = np.log(counts)
        
        n = len(scales)
        slope = safe_divide(
            n * np.sum(log_scales * log_counts) - np.sum(log_scales) * np.sum(log_counts),
            n * np.sum(log_scales**2) - np.sum(log_scales)**2,
            0.0
        )
        
        return float(-slope)
    
    def _empty_results(self):
        return {
            'd0': 1.0,
            'd1': 1.0,
            'd2': 1.0,
            'dimension_box': 1.0,
            'multifractality_index': 0.0,
            'lacunarity': 0.0,
            'lacunarity_curve': {},
            'complexity_score': 0.0,
            'log_scales': [],
            'log_counts': []
        }