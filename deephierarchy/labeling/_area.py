import numpy as np
from numba import njit


@njit
def compute_area(label_map: np.ndarray) -> np.ndarray:
    N = label_map.max() + 1

    res = np.zeros((N,), dtype=np.uint32)
    for li in range(label_map.shape[0]):
        for c in range(label_map.shape[1]):
            res[label_map[li, c]] += 1

    return res
