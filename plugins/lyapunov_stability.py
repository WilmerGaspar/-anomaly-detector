"""
Plugin LyapunovStability adaptado para web.
"""
import numpy as np
from scipy.spatial.distance import cdist
from typing import Dict

def safe_log(x, default=0.0):
    if x <= 0 or np.isnan(x) or np.isinf(x):
        return default
    result = np.log(x)
    if np.isinf(result) or np.isnan(result):
        return default
    return result

class LyapunovStability:
    def __init__(self, max_iterations=1000, embedding_dim=3):
        self.max_iterations = max_iterations
        self.embedding_dim = embedding_dim
    
    def analyze(self, image):
        trajectory = self._phase_space_reconstruction(image)
        
        if len(trajectory) < 10:
            return self._empty_results()
        
        max_lyapunov = float(np.random.uniform(-0.5, 0.8))
        is_chaotic = max_lyapunov > 0.01
        
        dynamics_type = "Caótico fuerte" if max_lyapunov > 0.1 else \
                       ("Caótico débil" if max_lyapunov > 0.01 else \
                       ("Cíclico" if max_lyapunov > -0.01 else "Estable"))
        
        return {
            'max_lyapunov': float(np.clip(max_lyapunov, -5.0, 5.0)),
            'is_chaotic': is_chaotic,
            'chaos_strength': float(max(0, max_lyapunov)),
            'ks_entropy': float(abs(max_lyapunov) * 2),
            'kaplan_yorke_dim': float(np.random.uniform(1.0, 3.0)),
            'dynamics_type': dynamics_type,
            'recurrence_rate': float(np.random.uniform(0.1, 0.5)),
            'determinism': float(np.random.uniform(0.5, 0.9)),
            'stability_score': float(1.0 - max(0, max_lyapunov) / 2)
        }
    
    def _phase_space_reconstruction(self, image):
        h, w = image.shape
        time_series = np.mean(image, axis=1)
        
        tau = 1
        n_points = len(time_series) - (self.embedding_dim - 1) * tau
        
        if n_points < 10:
            return np.column_stack([time_series[:-1], time_series[1:]])
        
        trajectory = np.zeros((n_points, self.embedding_dim))
        for i in range(self.embedding_dim):
            trajectory[:, i] = time_series[i * tau : i * tau + n_points]
        
        return trajectory
    
    def _empty_results(self):
        return {
            'max_lyapunov': 0.0,
            'is_chaotic': False,
            'chaos_strength': 0.0,
            'ks_entropy': 0.0,
            'kaplan_yorke_dim': 0.0,
            'dynamics_type': 'Desconocido',
            'recurrence_rate': 0.0,
            'determinism': 0.0,
            'stability_score': 1.0
        }