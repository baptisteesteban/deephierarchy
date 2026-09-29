import numpy as np
from numba import njit

from deephierarchy.core import Connectivity


class RAG:
    def __init__(self, adjacency_matrix: np.ndarray | None):
        self._adjacency_matrix = adjacency_matrix

    @property
    def adjacency_matrix(self) -> np.ndarray | None:
        return self._adjacency_matrix

    @property
    def num_nodes(self) -> int:
        return self._adjacency_matrix.shape[0] if self._adjacency_matrix is not None else 0


@njit
def build_rag_mat(label_map: np.ndarray, connectivity: np.ndarray) -> np.ndarray:
    N = label_map.max() + 1
    res = np.zeros((N, N), dtype=np.bool_)
    n_rows = label_map.shape[0]
    n_cols = label_map.shape[1]
    n_neighbors = connectivity.shape[0]

    for li in range(n_rows):
        for c in range(n_cols):
            cur = label_map[li, c]
            for k in range(n_neighbors):
                dl = connectivity[k, 0]
                dc = connectivity[k, 1]
                nl = li + dl
                nc = c + dc
                if 0 <= nl < n_rows and 0 <= nc < n_cols:
                    n_lbl = label_map[nl, nc]
                    if n_lbl != cur:
                        res[cur, n_lbl] = True
                        res[n_lbl, cur] = True

    return res


def build_rag(label_map: np.ndarray, connectivity: Connectivity) -> RAG:
    adjacency_matrix = build_rag_mat(label_map, connectivity.data)
    return RAG(adjacency_matrix)
