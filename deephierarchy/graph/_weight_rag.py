import numpy as np

from deephierarchy.graph import RAG


def weight_rag_dist_np(adjacency_matrix: np.ndarray, mean_value: np.ndarray) -> np.ndarray:
    res = np.full(adjacency_matrix.shape, -1, dtype=np.float64)

    for n1 in range(adjacency_matrix.shape[0]):
        for n2 in range(n1 + 1, adjacency_matrix.shape[1]):
            if adjacency_matrix[n1, n2]:
                res[n1, n2] = res[n2, n1] = np.linalg.norm(
                    mean_value[n1].astype(np.float64) - mean_value[n2].astype(np.float64)
                )

    return res


def weight_rag_dist(rag: RAG, mean_value: np.ndarray) -> np.ndarray:
    return weight_rag_dist_np(rag.adjacency_matrix, mean)


def weight_rag_weighted_area_dist_np(
    adjacency_matrix: np.ndarray, area: np.ndarray, sum: np.ndarray
) -> np.ndarray:
    res = np.full(adjacency_matrix.shape, -1, dtype=np.float64)

    for n1 in range(adjacency_matrix.shape[0]):
        for n2 in range(n1 + 1, adjacency_matrix.shape[1]):
            if adjacency_matrix[n1, n2]:
                union_area = area[n1] + area[n2]
                union_sum = sum[n1] + sum[n2]
                union_mean = union_sum / union_area
                n1_mean = sum[n1] / area[n1]
                n2_mean = sum[n2] / area[n2]
                res[n1, n2] = res[n2, n1] = area[n1] * np.linalg.norm(n1_mean - union_mean)
                +area[n2] * np.linalg.norm(n2_mean - union_mean)

    return res


def weight_rag_weighted_area_dist(rag: RAG, area: np.ndarray, sum: np.ndarray) -> np.ndarray:
    return weight_rag_weighted_area_dist_np(rag.adjacency_matrix, area, sum)
