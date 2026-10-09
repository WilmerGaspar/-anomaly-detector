"""Difracción del telescopio y geometría del cielo para corregir β y orientar las estructuras.

El espectro de potencia medido es el del cielo por MTF² (la respuesta del telescopio a cada
escala). Las imágenes de JWST están muestreadas casi al límite de difracción, así que la MTF
cae dentro del rango donde se ajusta β y lo empina. Medido con campos sintéticos pasados por
la MTF de una apertura de 6.5 m (región de 630 px):
  MIRI F2100W: β 2.50 -> 3.58 y 3.67 -> 4.6-4.8;  MIRI F1000W: 2.50 -> 2.84;  NIRCam F200W: 2.50 -> 2.8.
Corregido (P / MTF², solo hasta x = k/ν_c = 0.4): 2.43-2.65, 2.41-2.54 y 2.51-2.56.
La MTF es la de una apertura circular sin obstrucción: la de JWST (hexagonal, con espejo
secundario) y el muestreo del detector la bajan algo más, así que la corrección es por defecto.
"""
from __future__ import annotations

import re

import numpy as np

TELESCOPE_D_M = {"JWST": 6.5, "HST": 2.4}
X_MAX = 0.4                 # fracción de la frecuencia de corte hasta la que se ajusta (MTF² >= 0.25)
MIN_RANGE = 3.0             # rango mínimo k_max / k_min para que β tenga sentido
ARCSEC_PER_RAD = 206264.806


def telescope_key(telescope, instrument=""):
    t = str(telescope or "").upper()
    if "JWST" in t or "WEBB" in t:
        return "JWST"
    if "HST" in t or "HUBBLE" in t:
        return "HST"
    ins = str(instrument or "").upper()
    if ins in ("NIRCAM", "MIRI", "NIRISS", "NIRSPEC"):
        return "JWST"
    if ins in ("ACS", "WFC3", "WFPC2", "NICMOS", "STIS"):
        return "HST"
    return None


def wavelength_um(telescope, instrument, filt):
    """Longitud de onda central del filtro a partir de su nombre (F200W, F2100W, F814W, F125W…)."""
    tel = telescope_key(telescope, instrument)
    m = re.search(r"F(\d{3,4})", str(filt or "").upper())
    if not tel or not m:
        return None
    n = int(m.group(1))
    if tel == "JWST":                      # NIRCam/NIRISS/MIRI: centésimas de micra (F200W 2.00, F2100W 21.0)
        return n / 100.0
    ins = str(instrument or "").upper()
    if ins in ("WFC3", "NICMOS") and n < 200:
        return n / 100.0                   # WFC3/IR: F125W = 1.25 µm
    return n / 1000.0                      # ópticos: F814W = 0.814 µm


def cutoff_cycles_per_px(telescope, instrument, filt, pixel_scale_arcsec):
    """Frecuencia de corte de la difracción D/λ en ciclos por píxel, o None si falta algo."""
    tel = telescope_key(telescope, instrument)
    lam = wavelength_um(telescope, instrument, filt)
    try:
        pix = float(pixel_scale_arcsec)
    except (TypeError, ValueError):
        return None
    if not tel or not lam or not np.isfinite(pix) or pix <= 0:
        return None
    return TELESCOPE_D_M[tel] / (lam * 1e-6) * (pix / ARCSEC_PER_RAD)


def mtf(k_cycles_per_px, nu_c):
    """MTF de difracción de una apertura circular (0 a partir de la frecuencia de corte)."""
    x = np.clip(np.asarray(k_cycles_per_px, dtype=float) / float(nu_c), 0.0, 1.0)
    return (2.0 / np.pi) * (np.arccos(x) - x * np.sqrt(1.0 - x * x))


def describe(meta):
    """Lo que se sabe de la difracción para una región: para el JSON y la corrección."""
    lam = wavelength_um(meta.get("telescope"), meta.get("instrument"), meta.get("filter"))
    nu_c = cutoff_cycles_per_px(meta.get("telescope"), meta.get("instrument"), meta.get("filter"),
                                meta.get("pixel_scale_arcsec"))
    return {"telescope": telescope_key(meta.get("telescope"), meta.get("instrument")), "lambda_um": lam,
            "pixel_scale_arcsec": meta.get("pixel_scale_arcsec"), "nu_c_cycles_per_px": nu_c,
            "fwhm_px": (1.03 / nu_c) if nu_c else None, "x_max": X_MAX}


# ---------------------------------------------------------------- geometría del cielo

def wcs_info(header):
    """Escala de píxel (″) y cabecera WCS celeste (para el centro y las direcciones en el cielo)."""
    try:
        from astropy.wcs import WCS
        from astropy.wcs.utils import proj_plane_pixel_scales
        w = WCS(header).celestial
        if not w.has_celestial:
            return {}
        sc = proj_plane_pixel_scales(w) * 3600.0          # grados -> segundos de arco
        hdr = w.to_header()
        return {"pixel_scale_arcsec": float(np.sqrt(sc[0] * sc[1])),
                "wcs": {k: hdr[k] for k in hdr if isinstance(hdr[k], (int, float, str))}}
    except Exception:
        return {}


def sky_geometry(wcs, x0, y0, side, image_angles_deg=()):
    """Centro de la región (RA, Dec en grados) y, para cada ángulo de la imagen (desde +x hacia +y
    del array, en grados), su ángulo de posición en el cielo (este desde el norte, 0-180: son ejes)."""
    if not wcs:
        return {}
    try:
        from astropy.io import fits
        from astropy.wcs import WCS
        w = WCS(fits.Header(wcs))
        cx, cy = x0 + (side - 1) / 2.0, y0 + (side - 1) / 2.0
        ra, dec = (float(v) for v in w.pixel_to_world_values(cx, cy))
        out = {"center_ra_deg": ra, "center_dec_deg": dec, "pa_deg": []}
        for a in image_angles_deg:
            if a is None or not np.isfinite(a):
                out["pa_deg"].append(None)
                continue
            t = np.radians(a)
            r2, d2 = (float(v) for v in w.pixel_to_world_values(cx + 20 * np.cos(t), cy + 20 * np.sin(t)))
            d_east = ((r2 - ra + 180.0) % 360.0 - 180.0) * np.cos(np.radians(dec))      # sin salto en RA = 0
            d_north = d2 - dec
            out["pa_deg"].append(float(np.degrees(np.arctan2(d_east, d_north)) % 180.0))
        return out
    except Exception:
        return {}
