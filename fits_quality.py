"""SCI + ERR + DQ de productos JWST/HST i2d."""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np

def _first_array(hdu):
    data = getattr(hdu, "data", None)
    if data is None:
        return None
    arr = np.array(data)
    return arr if arr.size else None

def extract_layers(hdul) -> Dict:
    sci = err = dq = None
    sci_name = None
    for hdu in hdul:
        name = str(getattr(hdu, "name", "") or "").upper()
        arr = _first_array(hdu)
        if arr is None:
            continue
        if name in {"SCI", "SCIENCE"} or (sci is None and name in {"", "PRIMARY"} and arr.ndim >= 2):
            if name in {"SCI", "SCIENCE"} or sci is None:
                sci, sci_name = arr, name or "PRIMARY"
        if name in {"ERR", "ERROR", "RMS", "WHT"} and err is None:
            err = arr
        if name in {"DQ", "QUALITY", "MASK"} and dq is None:
            dq = arr
    if sci is None:
        for hdu in hdul:
            arr = _first_array(hdu)
            if arr is not None and np.asarray(arr).ndim >= 2:
                sci = arr
                sci_name = getattr(hdu, "name", "DATA")
                break
    return {"sci": sci, "err": err, "dq": dq, "sci_name": sci_name}

def _squeeze2(a):
    if a is None:
        return None
    a = np.squeeze(np.asarray(a))
    if a.ndim > 2:
        a = a[0]
    return a

def good_pixel_mask(sci, dq=None, err=None):
    sci = _squeeze2(sci)
    mask = np.isfinite(sci)
    if dq is not None:
        dq = _squeeze2(dq)
        if dq.shape == sci.shape:
            mask &= dq == 0
    if err is not None:
        err = _squeeze2(err)
        if err.shape == sci.shape:
            mask &= np.isfinite(err) & (err >= 0)
    return mask

def snr_weights(sci, err, mask):
    if err is None:
        return None
    err = _squeeze2(err)
    sci = _squeeze2(sci)
    if err.shape != sci.shape:
        return None
    w = np.zeros_like(sci, dtype=np.float32)
    good = mask & (err > 0)
    w[good] = np.clip(np.abs(sci[good]) / (err[good] + 1e-12), 0, 50)
    return w

def stretch_masked(sci, mask, p_lo=2.0, p_hi=98.0):
    sci = np.nan_to_num(_squeeze2(sci).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    vals = sci[mask] if mask is not None and mask.any() else sci.ravel()
    if vals.size == 0:
        return np.zeros_like(sci)
    lo, hi = np.percentile(vals, [p_lo, p_hi])
    out = np.zeros_like(sci) if hi <= lo else np.clip((sci - lo) / (hi - lo), 0.0, 1.0)
    if mask is not None:
        out = out.copy()
        out[~mask] = np.median(out[mask]) if mask.any() else 0.0
    return out

def apply_weights(image, weights):
    if weights is None:
        return image
    w = weights / (np.max(weights) + 1e-12)
    return np.clip(image * (0.35 + 0.65 * w), 0.0, 1.0)

def summarize_quality(mask, dq=None, err=None):
    n = int(mask.size)
    n_good = int(mask.sum())
    return {"has_err": err is not None, "has_dq": dq is not None, "n_pixels": n, "n_good": n_good, "good_fraction": float(n_good / n) if n else 0.0, "n_masked": n - n_good}
