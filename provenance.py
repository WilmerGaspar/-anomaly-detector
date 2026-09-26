"""Confianza de la fuente. Las fotos de prensa de NASA mienten de forma sistematica.

Regla: FITS de archivo o no se publica el candidato.
"""
from __future__ import annotations

from typing import Dict, Optional


PRESS_HINTS = (
    "hubbleSite",
    "hubblesite",
    "webbtelescope",
    "images-assets.nasa.gov",
    "photojournal.jpl.nasa.gov",
    "esawebb",
    "spacetelescope.org/images/",
)

SCIENCE_HINTS = (
    "mast.stsci.edu",
    "archive.stsci.edu",
    "almascience",
    "irsa.ipac",
    "heasarc",
    "archive.eso.org",
    "i2d.fits",
    "cal.fits",
)


def assess_provenance(
    filename: str,
    metadata: Optional[Dict] = None,
    source_url: str = "",
) -> Dict:
    meta = metadata or {}
    name = (filename or "").lower()
    url = (source_url or "").lower()
    is_fits = bool(meta.get("is_fits")) or name.endswith((".fits", ".fit", ".fits.gz"))
    instrument = str(meta.get("instrument") or "")
    reasons = []
    score = 0.15

    if is_fits:
        score += 0.45
        reasons.append("Hay contenedor FITS (pixeles + header).")
    else:
        reasons.append("JPEG/PNG: casi seguro stretch o composicion. No publicable.")

    if any(h in url or h in name for h in SCIENCE_HINTS):
        score += 0.25
        reasons.append("Huele a archivo cientifico (MAST/ALMA/IRSA/ESO).")
    if any(h in url or h in name for h in PRESS_HINTS):
        score -= 0.35
        reasons.append("Huele a galeria de prensa. Tratar como ilustracion.")

    if instrument:
        score += 0.1
        reasons.append(f"Instrumento declarado: {instrument}")

    ext = name.split(".")[-1]
    if ext in {"jpg", "jpeg", "png", "webp"}:
        score = min(score, 0.35)

    score = float(max(0.0, min(1.0, score)))
    if score >= 0.7:
        verdict = "usable_science"
    elif score >= 0.4:
        verdict = "exploratory_only"
    else:
        verdict = "reject_for_discovery"

    return {
        "trust_score": score,
        "verdict": verdict,
        "is_fits": is_fits,
        "reasons": reasons,
        "rule": "Sin FITS de archivo no hay candidato a ciencia en frontera.",
    }
