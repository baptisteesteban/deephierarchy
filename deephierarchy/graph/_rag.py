import numpy as np

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


def build_rag_mat(label_map: np.ndarray, connectivity: np.ndarray) -> np.ndarray:
    N = label_map.max() + 1
    res = np.zeros((N, N), dtype=bool)

    for li in range(label_map.shape[0]):
        for c in range(label_map.shape[1]):
            cur = label_map[li, c]
            for dl, dc in connectivity:
                nl, nc = li + dl, c + dc
                if (
                    nl >= 0
                    and nc >= 0
                    and nl < label_map.shape[0]
                    and nc < label_map.shape[1]
                    and (n_lbl := label_map[nl, nc]) != cur
                ):
                    res[cur, n_lbl] = res[n_lbl, cur] = True

    return res


def build_rag(label_map: np.ndarray, connectivity: Connectivity) -> RAG:
    adjacency_matrix = build_rag_mat(label_map, connectivity.data)
    return RAG(adjacency_matrix)
