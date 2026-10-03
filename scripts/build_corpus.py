#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import shutil
from mast_client import download_product, list_fits_products, search_observations
TARGETS = ["NGC 7023", "M16", "Orion Nebula", "Tarantula Nebula", "Carina Nebula", "Crab Nebula", "Rho Ophiuchi", "M51", "NGC 604", "Helix Nebula"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="corpus")
    ap.add_argument("--per-target", type=int, default=3)
    ap.add_argument("--missions", nargs="+", default=["HST", "JWST"])
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    saved = 0
    for target in TARGETS:
        print(">>", target)
        try:
            obs = search_observations(target, args.missions, radius_deg=0.1, limit=12)
        except Exception as exc:
            print(" query fail", exc); continue
        got = 0
        for row in obs:
            if got >= args.per_target: break
            try:
                prods = list_fits_products(row["obsid"])
            except Exception as exc:
                print(" products fail", exc); continue
            good = [p for p in prods if any(k in p["filename"].lower() for k in ("i2d", "drz", "drc")) and (p.get("size_mb") is None or p["size_mb"] <= 80)]
            if not good: continue
            prod = good[0]
            dest = out / ("%s_%s" % (target.replace(" ", "_"), prod["filename"]))
            if dest.exists():
                print(" have", dest.name); got += 1; saved += 1; continue
            try:
                path = download_product(prod["uri"], prod["filename"], max_mb=80)   # devuelve ruta
                shutil.move(path, dest)
                print(" ok", dest.name); got += 1; saved += 1
            except Exception as exc:
                print(" download fail", exc)
    print("saved", saved)

if __name__ == "__main__":
    main()
