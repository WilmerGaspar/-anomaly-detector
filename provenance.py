"""Confianza de la fuente y nivel de producto JWST/HST."""
from __future__ import annotations
from typing import Dict, Optional

PRESS_HINTS = (
    "hubblesite", "webbtelescope", "images-assets.nasa.gov",
    "photojournal.jpl.nasa.gov", "esawebb", "spacetelescope.org/images/",
)
SCIENCE_HINTS = (
    "mast.stsci.edu", "archive.stsci.edu", "almascience", "irsa.ipac",
    "heasarc", "archive.eso.org", "mast:jwst", "mast:hst", "i2d.fits", "cal.fits",
)
DETECTOR_LEVEL = (
    "uncal", "rateints", "rateint", "_rate.", "_raw", "dark", "flat", "sflat",
    "fflat", "traps", "bias", "refpix",
)
SCIENCE_LEVEL = ("i2d", "_drz", "_drc", "x1d", "s3d", "_calints", "_cal.")
SPEC_2D = ("nirspec", "nrs1", "nrs2", "niriss", "miri_lrs")


def _level(name: str) -> str:
    n = name.lower()
    if any(k in n for k in DETECTOR_LEVEL):
        return "detector"
    if "x1d" in n:
        return "spectrum_1d"
    if "s3d" in n:
        return "cube"
    if any(k in n for k in ("i2d", "_drz", "_drc")):
        return "image_calibrated"
    if any(k in n for k in SCIENCE_LEVEL):
        return "calibrated"
    return "unknown"


def assess_provenance(filename, metadata=None, source_url=""):
    meta = metadata or {}
    name = (filename or "").lower()
    url = (source_url or "").lower()
    is_fits = bool(meta.get("is_fits")) or name.endswith((".fits", ".fit", ".fits.gz"))
    instrument = str(meta.get("instrument") or "")
    reasons = []
    score = 0.15
    level = _level(name + " " + url)

    if is_fits:
        score += 0.45
        reasons.append("Hay contenedor FITS (pixeles + header).")
    else:
        reasons.append("JPEG/PNG: stretch o composicion. No publicable.")

    if any(h in url or h in name for h in SCIENCE_HINTS):
        score += 0.25
        reasons.append("Huele a archivo cientifico (MAST/ALMA/IRSA/ESO).")
    if any(h in url or h in name for h in PRESS_HINTS):
        score -= 0.35
        reasons.append("Huele a galeria de prensa.")

    if instrument:
        score += 0.1
        reasons.append("Instrumento declarado: %s" % instrument)

    if level == "detector":
        score = min(score, 0.45)
        reasons.append("Producto de detector (uncal/rate/rateints). No es mapa del cielo ni espectro 1D.")
    elif level == "spectrum_1d":
        reasons.append("Espectro 1D: usar libreria MIR, no scout de forma.")
    elif level == "image_calibrated":
        score = min(1.0, score + 0.1)
        reasons.append("Imagen calibrada (i2d/drz). Canal de morfologia.")

    inst_l = instrument.lower() + " " + name
    if any(k in inst_l for k in SPEC_2D) and level not in {"spectrum_1d", "cube", "image_calibrated"}:
        score = min(score, 0.5)
        reasons.append("Plano 2D de espectrografo. Busca x1d o una imagen NIRCam/MIRI i2d.")

    if name.endswith(("jpg", "jpeg", "png", "webp")) or (not is_fits):
        score = min(score, 0.35)

    score = float(max(0.0, min(1.0, score)))
    if level == "detector":
        verdict = "exploratory_only"
    elif score >= 0.7:
        verdict = "usable_science"
    elif score >= 0.4:
        verdict = "exploratory_only"
    else:
        verdict = "reject_for_discovery"

    return {
        "trust_score": score,
        "verdict": verdict,
        "is_fits": is_fits,
        "product_level": level,
        "reasons": reasons,
        "rule": "Sin imagen i2d/drz o espectro x1d no hay candidato a material.",
    }
