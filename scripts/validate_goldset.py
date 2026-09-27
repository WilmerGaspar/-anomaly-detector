#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
GOLD = json.loads((ROOT / "goldset" / "known_weird.json").read_text())

def main():
    if len(sys.argv) < 2:
        print(json.dumps(GOLD, indent=2, ensure_ascii=False)); return
    card = json.loads(Path(sys.argv[1]).read_text())
    name = (card.get("source") or {}).get("filename", "")
    fam = (card.get("morphology") or {}).get("family")
    for t in GOLD["targets"]:
        if t["name"].split()[0].lower() in name.lower() or t["mast_query"].split()[0].lower() in name.lower():
            if fam in (t.get("expect_not_family") or []):
                print("FAIL", t["name"], fam); sys.exit(1)
            print("OK", t["name"], fam); return
    print("no match in gold set for", name)

if __name__ == "__main__":
    main()
