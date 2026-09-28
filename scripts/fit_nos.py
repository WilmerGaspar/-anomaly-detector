#!/usr/bin/env python3
"""Entrena IsolationForest + EllipticEnvelope.
Uso:
  python scripts/fit_nos.py                 # fondo sintetico
  python scripts/fit_nos.py corpus.npy      # matriz N x 7 del barrido real
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nos_empirical import fit_nos, save_models, synthetic_sky_corpus
import numpy as np

def main():
    if len(sys.argv) > 1:
        X = np.load(sys.argv[1])
        src = sys.argv[1]
    else:
        X = synthetic_sky_corpus(600)
        src = "synthetic_sky"
    models = fit_nos(X)
    path = save_models(models)
    print("saved", path, "n=", len(X), "source=", src)

if __name__ == "__main__":
    main()
