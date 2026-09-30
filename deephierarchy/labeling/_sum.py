import numpy as np
from numba import njit


def compute_sum(label_map: np.ndarray, img: np.ndarray) -> np.ndarray:
    N = label_map.max() + 1
    TARGET_TYPE = np.uint32 if np.issubdtype(img.dtype, np.integer) else np.float32
    TARGET_SHAPE = (N,) if img.ndim == 2 else (N, img.shape[2])

    res = np.zeros(TARGET_SHAPE, dtype=TARGET_TYPE)
    for li in range(label_map.shape[0]):
        for c in range(label_map.shape[1]):
            res[label_map[li, c]] += img[li, c]

    return res
