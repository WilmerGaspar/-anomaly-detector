"""Cliente MAST: buscar objeto y listar FITS calibrados para elegir."""
from __future__ import annotations

from typing import Dict, List

PREFERRED = ("i2d.fits", "drc.fits", "drz.fits", "sci.fits", "cal.fits", "flt.fits")
BLOCK = ("jpg", "jpeg", "png", "thumb", "preview", "gif")


def search_observations(target: str, missions: List[str], radius_deg: float = 0.08, limit: int = 25) -> List[Dict]:
    from astroquery.mast import Observations

    table = Observations.query_object(target.strip(), radius=f"{radius_deg} deg")
    if table is None or len(table) == 0:
        return []
    missions_u = {m.upper() for m in missions}
    rows = []
    for rec in table:
        mission = str(rec.get("obs_collection") or rec.get("project") or "")
        if missions_u and mission.upper() not in missions_u:
            continue
        if str(rec.get("dataproduct_type") or "").lower() in {"timeseries", "catalog"}:
            continue
        rows.append({
            "obsid": str(rec.get("obsid") or rec.get("obs_id") or ""),
            "obs_id": str(rec.get("obs_id") or ""),
            "mission": mission,
            "instrument": str(rec.get("instrument_name") or rec.get("instrument") or ""),
            "filters": str(rec.get("filters") or rec.get("filter") or ""),
            "target": str(rec.get("target_name") or target),
        })
        if len(rows) >= limit:
            break
    return rows


def list_fits_products(obsid: str, max_mb: float = 80.0) -> List[Dict]:
    from astroquery.mast import Observations

    products = Observations.get_product_list(obsid)
    if products is None or len(products) == 0:
        return []
    out = []
    for rec in products:
        name = str(rec.get("productFilename") or rec.get("filename") or "")
        low = name.lower()
        if not low.endswith(".fits") and not low.endswith(".fits.gz"):
            continue
        if any(b in low for b in BLOCK):
            continue
        size = rec.get("size")
        try:
            size_mb = float(size) / (1024 * 1024) if size is not None else None
        except (TypeError, ValueError):
            size_mb = None
        if size_mb is not None and size_mb > max_mb:
            continue
        uri = str(rec.get("dataURI") or "")
        rank = 50
        for i, suf in enumerate(PREFERRED):
            if low.endswith(suf) or suf.replace(".fits", "") in low:
                rank = i
                break
        out.append({
            "filename": name,
            "product_type": str(rec.get("productType") or ""),
            "description": str(rec.get("description") or ""),
            "size_mb": size_mb,
            "uri": uri,
            "rank": rank,
        })
    out.sort(key=lambda r: (r["rank"], r["size_mb"] if r["size_mb"] is not None else 999))
    return out[:20]


def download_product(uri: str, filename: str, max_mb: float = 80.0):
    from astroquery.mast import Observations
    from pathlib import Path

    tmp = Path("/tmp/cms80_mast")
    tmp.mkdir(parents=True, exist_ok=True)
    dest = tmp / filename
    status = Observations.download_file(uri, local_path=str(dest))
    path = dest
    if isinstance(status, tuple):
        for item in status:
            if isinstance(item, str) and item.endswith((".fits", ".fits.gz")) and Path(item).exists():
                path = Path(item)
    if not Path(path).exists():
        found = list(tmp.glob("*.fits")) + list(tmp.glob("*.fits.gz"))
        if not found:
            raise RuntimeError(f"download fallo: {status}")
        path = found[-1]
    data = Path(path).read_bytes()
    if len(data) > max_mb * 1024 * 1024:
        raise RuntimeError(f"FITS > {max_mb:.0f} MB")
    return data
