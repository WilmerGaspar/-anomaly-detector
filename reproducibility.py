"""Versiones, semilla y hash del FITS."""
from __future__ import annotations
import hashlib
from typing import Optional

def fits_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def library_versions() -> dict:
    out = {}
    for name in ("numpy", "scipy", "astropy", "sklearn", "streamlit"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[name] = None
    return out

def stamp(raw: Optional[bytes] = None, seed: Optional[int] = None) -> dict:
    return {"schema": "cosmic-candidate.v0.1", "libraries": library_versions(), "rng_seed": seed, "fits_sha256": fits_sha256(raw) if raw else None}
