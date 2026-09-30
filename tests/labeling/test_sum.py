import numpy as np

from deephierarchy.labeling import compute_sum


def test_sum_grayscale():
    label_map = np.array([[0, 1, 2, 3], [1, 1, 2, 4], [1, 2, 2, 4]], dtype=int)
    img = np.array([[4, 5, 7, 3], [1, 2, 4, 5], [7, 7, 6, 4]], dtype=np.uint8)
    REF = np.array([4, 15, 24, 3, 9], dtype=np.uint32)
    res = compute_sum(label_map, img)
    assert res.dtype == REF.dtype
    assert np.all(res == REF)


def test_sum_color():
    label_map = np.array([[0, 1, 2, 3], [1, 1, 2, 4], [1, 2, 2, 4]], dtype=int)
    img = np.array(
        [
            [[2, 5, 7], [4, 2, 1], [1, 6, 9], [9, 4, 3]],
            [[4, 4, 7], [1, 3, 6], [5, 6, 7], [1, 0, 9]],
            [[9, 12, 7], [4, 3, 6], [4, 6, 7], [7, 1, 4]],
        ],
        dtype=np.uint8,
    )
    REF = np.array([[2, 5, 7], [18, 21, 21], [14, 21, 29], [9, 4, 3], [8, 1, 13]], dtype=np.uint32)
    res = compute_sum(label_map, img)
    assert res.dtype == REF.dtype
    assert np.all(res == REF)
