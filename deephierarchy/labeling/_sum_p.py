import numpy as np
from numba import njit


@njit
def compute_sum_p(label_map: np.ndarray) -> np.ndarray:
    N = label_map.max() + 1

    res = np.zeros((N, 2), dtype=float)
    for li in range(label_map.shape[0]):
        for c in range(label_map.shape[1]):
            res[label_map[li, c], 0] += li
            res[label_map[li, c], 1] += c

    return res
