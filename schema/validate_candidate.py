#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path

def validate(card: dict) -> list:
    errors = []
    if card.get("schema") != "cosmic-candidate.v0.1":
        errors.append("schema != cosmic-candidate.v0.1")
    if "source" not in card or "filename" not in card.get("source", {}):
        errors.append("falta source.filename")
    if "morphology" not in card or "is_candidate" not in card.get("morphology", {}):
        errors.append("falta morphology.is_candidate")
    if "spectrum" not in card:
        errors.append("falta spectrum")
    elif card["spectrum"].get("status") not in {"missing", "attached", "identified", "mir_unknown"}:
        errors.append("spectrum.status invalido")
    if "nos_morfoligical" in card:
        errors.append("typo nos_morfoligical; usar nos_morphological")
    return errors

def main():
    if len(sys.argv) < 2:
        print("uso: python schema/validate_candidate.py archivo.json"); sys.exit(2)
    card = json.loads(Path(sys.argv[1]).read_text())
    errs = validate(card)
    if errs:
        print("INVALID")
        [print(" -", e) for e in errs]
        sys.exit(1)
    print("OK", card.get("source", {}).get("filename"))

if __name__ == "__main__":
    main()
