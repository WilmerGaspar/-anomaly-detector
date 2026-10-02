"""Cliente MAST: busqueda con conteos reales, procedencia completa y descarga segura.

Cambios (ver CHANGES.md):
- search_observations_detailed devuelve cuantas observaciones trajo MAST, cuantas
  son de la mision elegida y cuantas son imagenes, ademas de las filas.
- Cada fila lleva su procedencia: programa, PI, titulo, fecha, exposicion, nivel
  de calibracion y URLs reales devueltas por MAST.
- x1d (tabla espectral) y s3d (cubo 3D) ya no se recomiendan: no son imagenes 2D.
- download_product verifica el tamano ANTES de descargar y ya no recurre a
  "cualquier FITS que haya en /tmp" si la descarga falla.
"""
from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

IMAGE_SUFFIXES = ("i2d.fits", "drc.fits", "drz.fits", "cal.fits", "flc.fits", "flt.fits", "sci.fits")
NOT_IMAGE_SUFFIXES = ("x1d.fits", "x1dints.fits", "s3d.fits", "c1d.fits", "asn.fits", "spt.fits")
# Detectores de espectrografos JWST: sus _cal/_rate son planos 2D de espectro (o
# imagenes de adquisicion para apuntar), no imagenes del cielo.
SPECTRO_DETECTORS = ("_nrs1", "_nrs2", "mirifushort", "mirifulong")
MISSION_ALIASES = {
    "HST": {"HST", "HLA"},
    "JWST": {"JWST"},
    "ROMAN": {"ROMAN", "RST", "NGRST"},
    "HLSP": {"HLSP"},
}
TIMEOUT_SEC = 45
MAX_CLOUD_MB = 400.0
MAST_DOWNLOAD = "https://mast.stsci.edu/api/v0.1/Download/file?uri="
MAST_ACK = ("Datos obtenidos del Mikulski Archive for Space Telescopes (MAST) del Space Telescope "
            "Science Institute. Cite también el programa (ID de propuesta) indicado en cada fila.")

CURATED = {
    "JWST": ["NGC 7023", "M16", "Orion Nebula", "Stephan's Quintet", "SMACS 0723", "NGC 3324", "Cartwheel Galaxy", "Jupiter", "PDS 70"],
    "HST": ["M16", "M51", "M42", "NGC 7023", "Crab Nebula", "Hoag Object", "Eta Carinae", "Helix Nebula", "Red Square Nebula", "HH 30"],
    "ROMAN": ["LMC", "SMC", "M31"],
    "HLSP": ["HUDF", "GOODS-S", "COSMOS", "CANDELS"],
}

NOTES = {
    "M16": "Pilares de la Creación. Busca i2d de NIRCam o drz de WFC3.",
    "M51": "Galaxia espiral. Buen control de estructura conocida.",
    "Crab Nebula": "Filamentos. Control positivo.",
    "Hoag Object": "Galaxia anular.",
    "Red Square Nebula": "Simetría marcada. Cuidado con los spikes de difracción.",
    "NGC 7023": "Nebulosa de reflexión con polvo.",
    "Stephan's Quintet": "Grupo de galaxias en interacción.",
    "SMACS 0723": "Lente gravitacional. Los arcos no son granos.",
    "Jupiter": "Planeta. Útil para probar el cargador, no para morfología del ISM.",
    "PDS 70": "Disco protoplanetario.",
    "LMC": "Campo amplio. Elige un cal o i2d, no un mosaico entero.",
    "SMC": "Campo amplio.",
    "M31": "Andrómeda. Trabaja con un recorte.",
    "HUDF": "Campo ultraprofundo de Hubble.",
    "GOODS-S": "Survey profundo.",
    "COSMOS": "Survey de campo amplio.",
}


def _obs():
    from astroquery.mast import Observations
    try:
        Observations.TIMEOUT = TIMEOUT_SEC
    except Exception:
        pass
    return Observations


def _get(rec, key, default=None):
    try:
        v = rec[key]
    except Exception:
        return default
    try:
        if hasattr(v, "mask") and bool(v.mask):
            return default
    except Exception:
        pass
    return default if v is None else v


def _mission_ok(raw, selected):
    raw_u = (raw or "").upper()
    for sel in selected:
        aliases = {a.upper() for a in MISSION_ALIASES.get(sel.upper(), {sel})}
        if raw_u in aliases or sel.upper() in raw_u:
            return True
    return False


def _mjd_to_date(mjd):
    try:
        return (_dt.datetime(1858, 11, 17) + _dt.timedelta(days=float(mjd))).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OverflowError):
        return ""


def _num(v, nd=None):
    try:
        x = float(v)
        return round(x, nd) if nd is not None else x
    except (TypeError, ValueError):
        return None


def is_image_product(name):
    low = (name or "").lower()
    if any(k in low for k in ("uncal", "_rate.", "rateints", "_raw.")):
        return False          # 'uncal.fits' termina en 'cal.fits': excluir antes
    if any(low.endswith(s) or low.endswith(s + ".gz") for s in NOT_IMAGE_SUFFIXES):
        return False
    if any(k in low for k in SPECTRO_DETECTORS):
        return False
    return any(low.endswith(s) or low.endswith(s + ".gz") for s in IMAGE_SUFFIXES)


def _hint(name, size_mb=None):
    low = (name or "").lower()
    if size_mb is not None and size_mb > MAX_CLOUD_MB:
        return "demasiado grande (>%d MB)" % int(MAX_CLOUD_MB)
    if "skycell" in low or "all_drc" in low:
        return "mosaico enorme"
    if any(s in low for s in ("x1d", "c1d")):
        return "espectro 1D (tabla, no imagen)"
    if "s3d" in low:
        return "cubo 3D (no imagen 2D)"
    if "uncal" in low or "_raw" in low:
        return "crudo del detector"
    if any(k in low for k in SPECTRO_DETECTORS):
        return "plano de espectrógrafo o adquisición (no imagen del cielo)"
    if "rateints" in low or "_rate." in low:
        return "cuentas/s del detector (nivel 2a)"
    if any(s in low for s in ("i2d", "_drz", "_drc")):
        return "imagen calibrada y combinada (recomendada)"
    if "_cal" in low:
        return "exposición calibrada"
    if "_flt" in low or "_flc" in low:
        return "exposición calibrada (HST)"
    return ""


def _row(rec, fallback_target=""):
    jpeg = str(_get(rec, "jpegURL", "") or "")
    if jpeg.startswith("/"):
        jpeg = "https://mast.stsci.edu" + jpeg
    return {
        "obsid": str(_get(rec, "obsid", "")),
        "obs_id": str(_get(rec, "obs_id", "")),
        "mission": str(_get(rec, "obs_collection", "") or _get(rec, "project", "")),
        "instrument": str(_get(rec, "instrument_name", "")),
        "filters": str(_get(rec, "filters", "")),
        "target": str(_get(rec, "target_name", "") or fallback_target),
        "s_ra": _num(_get(rec, "s_ra"), 5),
        "s_dec": _num(_get(rec, "s_dec"), 5),
        "t_exptime": _num(_get(rec, "t_exptime"), 1),
        "date_obs": _mjd_to_date(_get(rec, "t_min")),
        "proposal_id": str(_get(rec, "proposal_id", "")),
        "proposal_pi": str(_get(rec, "proposal_pi", "")),
        "obs_title": str(_get(rec, "obs_title", "")),
        "calib_level": _get(rec, "calib_level"),
        "intent": str(_get(rec, "intentType", "")),
        "data_rights": str(_get(rec, "dataRights", "")),
        "jpeg_url": jpeg if jpeg.startswith("http") else "",
        "data_url": str(_get(rec, "dataURL", "") or ""),
        "dataproduct_type": str(_get(rec, "dataproduct_type", "")).lower(),
    }


def search_observations_detailed(target, missions, radius_deg=0.12, limit=500, images_only=True):
    """Devuelve filas + conteos reales para mostrar al usuario."""
    Observations = _obs()
    last, table = None, None
    for _ in range(2):
        try:
            table = Observations.query_object(target.strip(), radius="%s deg" % radius_deg)
            last = None
            break
        except Exception as exc:
            last = exc
    if last is not None:
        raise RuntimeError("MAST no respondió (timeout o red). Reintenta o carga un FITS local. Detalle: %s" % last)
    out = {"target": target, "radius_deg": radius_deg, "missions": list(missions or []),
           "n_total": 0, "n_mission": 0, "n_images": 0, "n_shown": 0, "truncated": False, "rows": []}
    if table is None or len(table) == 0:
        return out
    out["n_total"] = int(len(table))
    for rec in table:
        mission = str(_get(rec, "obs_collection", "") or _get(rec, "project", ""))
        if missions and not _mission_ok(mission, missions):
            continue
        out["n_mission"] += 1
        dtype = str(_get(rec, "dataproduct_type", "")).lower()
        if images_only and dtype != "image":
            continue
        out["n_images"] += 1
        if len(out["rows"]) < limit:
            out["rows"].append(_row(rec, target))
    out["n_shown"] = len(out["rows"])
    out["truncated"] = out["n_shown"] < out["n_images"]
    return out


def search_observations(target, missions, radius_deg=0.12, limit=40):
    """Compatibilidad con la version anterior: solo las filas."""
    return search_observations_detailed(target, missions, radius_deg, limit, images_only=False)["rows"]


def list_fits_products(obsid, max_mb=None):
    Observations = _obs()
    last, products = None, None
    for _ in range(2):
        try:
            products = Observations.get_product_list(obsid)
            last = None
            break
        except Exception as exc:
            last = exc
    if last is not None:
        raise RuntimeError("MAST no respondió al listar los FITS. Detalle: %s" % last)
    if products is None or len(products) == 0:
        return []
    out, seen = [], set()
    for rec in products:
        name = str(_get(rec, "productFilename", "") or _get(rec, "filename", ""))
        low = name.lower()
        if not low.endswith((".fits", ".fits.gz")) or name in seen:
            continue
        seen.add(name)
        size_mb = _num(_get(rec, "size"))
        size_mb = size_mb / (1024 * 1024) if size_mb is not None else None
        if max_mb is not None and size_mb is not None and size_mb > max_mb:
            continue
        image = is_image_product(name)
        rank = 50
        if image:
            rank = next((i for i, s in enumerate(IMAGE_SUFFIXES) if low.endswith(s) or low.endswith(s + ".gz")), 40)
        if any(k in low for k in ("uncal", "rateints", "_rate.")):
            rank = 90
        too_big = bool(size_mb is not None and size_mb > MAX_CLOUD_MB)
        if "skycell" in low or too_big:
            rank = 95
        uri = str(_get(rec, "dataURI", "") or "")
        out.append({
            "filename": name,
            "product_type": str(_get(rec, "productType", "")),
            "calib_level": _get(rec, "calib_level"),
            "description": str(_get(rec, "description", "")),
            "size_mb": round(size_mb, 1) if size_mb is not None else None,
            "uri": uri,
            "download_url": (MAST_DOWNLOAD + uri) if uri else "",
            "rank": rank,
            "is_image": image,
            "hint": _hint(name, size_mb),
            "too_big": too_big,
        })
    out.sort(key=lambda r: (r["rank"], 9999 if r["size_mb"] is None else r["size_mb"]))
    return out


def _safe_name(filename):
    return re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).name) or "product.fits"


def download_product(uri, filename, max_mb=MAX_CLOUD_MB, size_mb=None):
    if size_mb is not None and size_mb > max_mb:
        raise RuntimeError("El archivo pesa %.0f MB y el tope es %.0f MB. Elige un cal/flt más pequeño." % (size_mb, max_mb))
    Observations = _obs()
    tmp = Path("/tmp/cms80_mast")
    tmp.mkdir(parents=True, exist_ok=True)
    dest = tmp / _safe_name(filename)
    result = Observations.download_file(uri, local_path=str(dest))
    status = result[0] if isinstance(result, tuple) and result else result
    if str(status).upper() not in ("COMPLETE", "SKIPPED") or not dest.exists():
        raise RuntimeError("La descarga de %s falló (estado: %s)." % (filename, result))
    if dest.stat().st_size > max_mb * 1024 * 1024:
        raise RuntimeError("El FITS pesa %.0f MB y el tope es %.0f MB." % (dest.stat().st_size / 1048576, max_mb))
    return dest.read_bytes()


def download_url(url, max_mb=MAX_CLOUD_MB, timeout=TIMEOUT_SEC):
    """Descarga un archivo desde una URL directa con tope de tamano."""
    import urllib.request
    if not url.lower().startswith(("https://", "http://")):
        raise ValueError("La URL debe empezar con https:// o http://")
    cap = int(max_mb * 1024 * 1024)
    req = urllib.request.Request(url, headers={"User-Agent": "CMS-80/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        length = resp.headers.get("Content-Length")
        if length and int(length) > cap:
            raise RuntimeError("El archivo pesa %.0f MB y el tope es %.0f MB." % (int(length) / 1048576, max_mb))
        data = resp.read(cap + 1)
    if len(data) > cap:
        raise RuntimeError("El archivo supera el tope de %.0f MB." % max_mb)
    return data
