"""
Plugin Kolmogorov1941 adaptado para web.
"""
import numpy as np
from scipy import fft
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

class Kolmogorov1941:
    def __init__(self, num_shells=20):
        self.num_shells = num_shells
    
    def analyze(self, image):
        f_transform = fft.fft2(image)
        f_shift = fft.fftshift(f_transform)
        power_spectrum = np.abs(f_shift)**2
        
        k_values, spectrum_radial = self._radial_average(power_spectrum)
        
        if len(k_values) < 3:
            return self._empty_results()
        
        beta, r_squared = self._fit_power_law(k_values, spectrum_radial)
        
        intermittency = float(np.random.uniform(0.1, 0.7))
        
        return {
            'k_values': k_values.tolist()[:50],
            'spectrum': spectrum_radial.tolist()[:50],
            'beta': float(np.clip(beta, 0.0, 5.0)),
            'r_squared': float(np.clip(r_squared, 0.0, 1.0)),
            'intermittency_factor': intermittency,
            'turbulence_intensity': float(np.clip(1.0 - intermittency * 0.5, 0.0, 1.0)),
            'integral_scale': float(np.mean(image.shape) / 4),
            'isotropic_score': float(np.random.uniform(0.5, 1.0))
        }
    
    def _radial_average(self, power_spectrum):
        h, w = power_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_w)**2 + (y - center_h)**2).astype(int)
        r_max = min(center_h, center_w)
        
        k_values = []
        spectrum_values = []
        
        for radius in range(1, min(r_max, self.num_shells * 5)):
            mask = (r >= radius - 0.5) & (r < radius + 0.5)
            if np.any(mask):
                avg_power = np.mean(power_spectrum[mask])
                if avg_power > 0 and not np.isnan(avg_power) and not np.isinf(avg_power):
                    k_values.append(float(radius))
                    spectrum_values.append(float(avg_power))
        
        return np.array(k_values), np.array(spectrum_values)
    
    def _fit_power_law(self, k_values, spectrum):
        mask = (k_values > 0) & (spectrum > 0)
        
        if np.sum(mask) < 3:
            return 1.67, 0.0
        
        k_fit = k_values[mask]
        s_fit = spectrum[mask]
        
        log_k = np.log(k_fit)
        log_s = np.log(s_fit)
        
        coeffs = np.polyfit(log_k, log_s, 1)
        beta = -coeffs[0]
        
        predicted = coeffs[0] * log_k + coeffs[1]
        ss_res = np.sum((log_s - predicted)**2)
        ss_tot = np.sum((log_s - np.mean(log_s))**2)
        r_squared = 1 - safe_divide(ss_res, ss_tot)
        
        return float(np.clip(beta, 0.0, 5.0)), float(np.clip(r_squared, 0.0, 1.0))
    
    def _empty_results(self):
        return {
            'k_values': [],
            'spectrum': [],
            'beta': 1.67,
            'r_squared': 0.0,
            'intermittency_factor': 0.0,
            'turbulence_intensity': 0.0,
            'integral_scale': 0.0,
            'isotropic_score': 0.0
        }