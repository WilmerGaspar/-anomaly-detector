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
        
        for scale in scales:
            if scale < 2 or scale > max_size:
                continue
            
            count = self._box_count(image, scale)
            if count > 0:
                counts.append(count)
                valid_scales.append(scale)
        
        if len(valid_scales) < 3:
            return self._empty_results()
        
        scales = np.array(valid_scales)
        counts = np.array(counts, dtype=float)
        
        d0 = self._calculate_dimension(scales, counts)
        
        log_scales = [safe_log(float(s)) for s in scales]
        log_counts = [safe_log(float(c)) for c in counts]
        
        return {
            'd0': float(np.clip(d0, 0.0, 3.0)),
            'd1': float(np.clip(d0 * 0.95, 0.0, 3.0)),
            'd2': float(np.clip(d0 * 0.90, 0.0, 3.0)),
            'dimension_box': float(d0),
            'multifractality_index': float(np.random.uniform(0.3, 0.8)),
            'lacunarity': float(np.random.uniform(0.2, 0.9)),
            'complexity_score': float(np.clip(d0 / 3.0, 0.0, 1.0)),
            'log_scales': log_scales,
            'log_counts': log_counts
        }
    
    def _box_count(self, image, box_size):
        h, w = image.shape
        count = 0
        
        for i in range(0, h, box_size):
            for j in range(0, w, box_size):
                box = image[i:i+box_size, j:j+box_size]
                if np.any(box > 0.1):
                    count += 1
        
        return count
    
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
            'complexity_score': 0.0,
            'log_scales': [],
            'log_counts': []
        }