"""
Topologia de la mascara de estructura: numeros de Betti medidos.

Antes betti_0, betti_1 y la "entropia topologica" eran numeros al azar (np.random.uniform):
cambiaban en cada ejecucion y movian las familias 'filament', 'lattice' y 'compact'.

Ahora, sobre la mascara de pixeles > media + 0.5 sigma (el mismo umbral que antes usaba
para escoger puntos):
- betti_0: componentes conexas (8-vecinos) con area >= min_area;
- betti_1: huecos, componentes del fondo (4-vecinos, dual de 8) que no tocan el borde,
  con area >= min_area;
- min_area = max(16 px, 0.05 % de la region): sin el, el ruido da miles de motas de 1-2 px.
No es homologia persistente completa (un solo umbral); el nombre del plugin se conserva.
"""
import numpy as np
from scipy import ndimage


class PersistentHomology:
    def __init__(self, min_area=16, min_area_frac=0.0005):
        self.min_area = min_area
        self.min_area_frac = min_area_frac

    def analyze(self, image):
        a = np.nan_to_num(np.asarray(image, dtype=float), nan=0.0)
        if a.size == 0 or a.std() == 0:
            return self._empty_results()
        mask = a > a.mean() + 0.5 * a.std()
        min_area = max(self.min_area, int(self.min_area_frac * a.size))

        lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
        sizes = np.bincount(lab.ravel())[1:] if n else np.array([], dtype=int)
        comp = sizes[sizes >= min_area]

        blab, nb = ndimage.label(~mask)
        bsizes = np.bincount(blab.ravel())[1:] if nb else np.array([], dtype=int)
        edge = np.unique(np.concatenate([blab[0], blab[-1], blab[:, 0], blab[:, -1]]))
        inner = np.ones(nb, dtype=bool)
        inner[edge[edge > 0] - 1] = False
        holes = bsizes[inner & (bsizes >= min_area)]

        b0, b1 = int(len(comp)), int(len(holes))
        if b0:
            p = comp / comp.sum()
            entropy = float(-(p * np.log2(p)).sum())
            connectivity = float(comp.max() / comp.sum())
        else:
            entropy, connectivity = 0.0, 0.0
        return {
            'betti_0': b0,
            'betti_1': b1,
            'betti_numbers': [b0, b1],
            'euler_characteristic': b0 - b1,
            # Entropia (bits) del reparto de area entre componentes: 0 = una sola pieza.
            'topological_entropy': entropy,
            'n_components_all': int(n),
            'min_area_px': int(min_area),
            'complexity_score': float(np.clip((b0 + b1) / 20.0, 0.0, 1.0)),
            # Fraccion del area de estructura en la componente mayor.
            'connectivity_index': connectivity,
            'topology_type': 'Vacío' if b0 == 0 else ('Complejo' if b1 > 3 else 'Simple'),
        }

    def _empty_results(self):
        return {
            'betti_0': 0,
            'betti_1': 0,
            'betti_numbers': [0, 0],
            'euler_characteristic': 0,
            'topological_entropy': 0.0,
            'n_components_all': 0,
            'min_area_px': 0,
            'complexity_score': 0.0,
            'connectivity_index': 0.0,
            'topology_type': 'Vacío'
        }
