"""Cliente MAST para HST / JWST / Roman. Lista todos los FITS."""
from __future__ import annotations

PREFERRED = ("i2d.fits", "drc.fits", "drz.fits", "x1d.fits", "s3d.fits", "cal.fits", "flt.fits", "sci.fits")
MISSION_ALIASES = {
    "HST": {"HST", "HLA"},
    "JWST": {"JWST"},
    "ROMAN": {"ROMAN", "RST", "NGRST"},
    "HLSP": {"HLSP"},
}

def _mission_ok(raw, selected):
    raw_u = (raw or "").upper()
    for sel in selected:
        aliases = {a.upper() for a in MISSION_ALIASES.get(sel.upper(), {sel})}
        if raw_u in aliases or sel.upper() in raw_u:
            return True
    return False

def _hint(name):
    low = (name or "").lower()
    if any(s in low for s in ("i2d", "_drz", "_drc")):
        return "imagen calibrada"
    if "x1d" in low:
        return "espectro 1D"
    if "s3d" in low:
        return "cubo 3D"
    if "uncal" in low or "_raw" in low:
        return "CRUDO detector — no usar para materiales"
    if "rate" in low or "_cal" in low:
        return "calibrado intermedio"
    return ""

def search_observations(target, missions, radius_deg=0.12, limit=80):
    from astroquery.mast import Observations
    table = Observations.query_object(target.strip(), radius="%s deg" % radius_deg)
    if table is None or len(table) == 0:
        return []
    rows = []
    for rec in table:
        mission = str(rec.get("obs_collection") or rec.get("project") or "")
        if missions and not _mission_ok(mission, missions):
            continue
        dtype = str(rec.get("dataproduct_type") or "").lower()
        if dtype in {"timeseries", "catalog"}:
            continue
        jpeg = str(rec.get("jpegURL") or "")
        if jpeg.startswith("/") and not jpeg.startswith("http"):
            jpeg = "https://mast.stsci.edu" + jpeg
        rows.append({
            "obsid": str(rec.get("obsid") or ""),
            "obs_id": str(rec.get("obs_id") or ""),
            "mission": mission,
            "instrument": str(rec.get("instrument_name") or ""),
            "filters": str(rec.get("filters") or ""),
            "target": str(rec.get("target_name") or target),
            "s_ra": rec.get("s_ra"),
            "s_dec": rec.get("s_dec"),
            "t_exptime": rec.get("t_exptime"),
            "jpeg_url": jpeg if jpeg.startswith("http") else "",
            "dataproduct_type": dtype,
        })
        if len(rows) >= limit:
            break
    return rows

def list_fits_products(obsid, max_mb=None):
    from astroquery.mast import Observations
    products = Observations.get_product_list(obsid)
    if products is None or len(products) == 0:
        return []
    out = []
    seen = set()
    for rec in products:
        name = str(rec.get("productFilename") or rec.get("filename") or "")
        low = name.lower()
        if not low.endswith((".fits", ".fits.gz")):
            continue
        if name in seen:
            continue
        seen.add(name)
        size = rec.get("size")
        try:
            size_mb = float(size) / (1024 * 1024) if size is not None else None
        except (TypeError, ValueError):
            size_mb = None
        if max_mb is not None and size_mb is not None and size_mb > max_mb:
            continue
        uri = str(rec.get("dataURI") or "")
        rank = 80
        for i, suf in enumerate(PREFERRED):
            if low.endswith(suf) or suf.replace(".fits", "") in low:
                rank = i
                break
        if "uncal" in low:
            rank = 90
        out.append({
            "filename": name,
            "product_type": str(rec.get("productType") or ""),
            "description": str(rec.get("description") or rec.get("productSubGroupDescription") or ""),
            "size_mb": size_mb,
            "uri": uri,
            "rank": rank,
            "hint": _hint(name),
        })
    out.sort(key=lambda r: (r["rank"], 9999 if r["size_mb"] is None else r["size_mb"]))
    return out

def download_product(uri, filename, max_mb=200.0):
    from astroquery.mast import Observations
    from pathlib import Path
    tmp = Path("/tmp/cms80_mast")
    tmp.mkdir(parents=True, exist_ok=True)
    dest = tmp / filename
    status = Observations.download_file(uri, local_path=str(dest))
    path = dest
    if isinstance(status, tuple):
        for item in status:
            if isinstance(item, str) and Path(item).exists():
                path = Path(item)
    if not Path(path).exists():
        found = list(tmp.glob("*.fits")) + list(tmp.glob("*.fits.gz"))
        if not found:
            raise RuntimeError("download fallo: %s" % (status,))
        path = found[-1]
    data = Path(path).read_bytes()
    if len(data) > max_mb * 1024 * 1024:
        raise RuntimeError("FITS > %.0f MB. Elige uno mas chico (i2d/x1d)." % max_mb)
    return data
