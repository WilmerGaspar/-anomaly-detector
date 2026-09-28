"""Teoria de grafos sobre mosaicos del campo.

Nodos = tiles. Arista si son vecinos y el contraste es bajo (misma fase).
Metricas: componentes, clustering local, gap espectral del Laplaciano.
"""
from __future__ import annotations
import numpy as np

def analyze_graph(image, n=8):
    img = np.asarray(image, dtype=np.float64)
    if img.ndim > 2:
        img = img.mean(axis=2)
    h, w = img.shape
    th, tw = max(4, h // n), max(4, w // n)
    means = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            patch = img[i*th:(i+1)*th, j*tw:(j+1)*tw]
            means[i, j] = float(np.nanmean(patch)) if patch.size else 0.0
    nodes = n * n
    adj = np.zeros((nodes, nodes), dtype=float)
    def idx(i, j):
        return i * n + j
    thr = float(np.nanstd(means) * 0.75 + 1e-6)
    for i in range(n):
        for j in range(n):
            for di, dj in ((1, 0), (0, 1)):
                ii, jj = i + di, j + dj
                if ii >= n or jj >= n:
                    continue
                d = abs(means[i, j] - means[ii, jj])
                if d <= thr:
                    a, b = idx(i, j), idx(ii, jj)
                    wgt = 1.0 - d / (thr + 1e-9)
                    adj[a, b] = adj[b, a] = wgt
    deg = adj.sum(axis=1)
    n_edges = int((adj > 0).sum() // 2)
    # componentes via BFS
    seen = np.zeros(nodes, dtype=bool)
    comps = 0
    largest = 0
    for s in range(nodes):
        if seen[s]:
            continue
        comps += 1
        stack = [s]
        seen[s] = True
        size = 0
        while stack:
            u = stack.pop()
            size += 1
            for v in np.where(adj[u] > 0)[0]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(int(v))
        largest = max(largest, size)
    # clustering medio (triangulos / posibles)
    clust = []
    for u in range(nodes):
        nbr = np.where(adj[u] > 0)[0]
        k = len(nbr)
        if k < 2:
            continue
        links = 0
        for a in range(k):
            for b in range(a + 1, k):
                if adj[nbr[a], nbr[b]] > 0:
                    links += 1
        clust.append(2.0 * links / (k * (k - 1)))
    clustering = float(np.mean(clust)) if clust else 0.0
    # gap espectral
    gap = float("nan")
    try:
        d = np.diag(deg)
        L = d - adj
        eig = np.sort(np.linalg.eigvalsh(L))
        if len(eig) >= 2:
            gap = float(eig[1])
    except Exception:
        pass
    return {
        "n_nodes": int(nodes),
        "n_edges": n_edges,
        "n_components": int(comps),
        "largest_component": int(largest),
        "mean_degree": float(deg.mean()) if nodes else 0.0,
        "clustering": clustering,
        "spectral_gap": gap,
        "threshold": thr,
        "adjacency": adj.tolist(),
        "note": "Pocas componentes + clustering alto = dominios cohesivos (filamento/agregado). Muchas componentes = campo fragmentado.",
    }
