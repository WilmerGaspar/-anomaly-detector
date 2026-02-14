"""
Plugins matemáticos para versión web.
"""

from .fractal_base import FractalBase
from .kolmogorov_1941 import Kolmogorov1941
from .lyapunov_stability import LyapunovStability
from .persistent_homology import PersistentHomology
from .renormalization_group import RenormalizationGroup

__all__ = [
    'FractalBase',
    'Kolmogorov1941',
    'LyapunovStability',
    'PersistentHomology',
    'RenormalizationGroup'
]