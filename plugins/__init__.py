"""Plugins de descriptores de imagen. Todos calculan métricas reales (sin placeholders)."""

from .fractal_base import FractalBase
from .kolmogorov_1941 import Kolmogorov1941
from .lyapunov_stability import LyapunovStability
from .persistent_homology import PersistentHomology
from .renormalization_group import RenormalizationGroup
from .anisotropy import calculate_anisotropy
from .entropy import calculate_entropy
from .periodicity import analyze_periodicity

__all__ = [
    "FractalBase",
    "Kolmogorov1941",
    "LyapunovStability",
    "PersistentHomology",
    "RenormalizationGroup",
    "calculate_anisotropy",
    "calculate_entropy",
    "analyze_periodicity",
]
