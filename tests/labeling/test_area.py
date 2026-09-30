import numpy as np

from deephierarchy.labeling import compute_area


def test_area():
    label_map = np.array([[0, 1, 2, 3], [1, 1, 2, 4], [1, 2, 2, 4]], dtype=int)
    REF = np.array([1, 4, 4, 1, 2], dtype=int)
    res = compute_area(label_map)
    assert np.all(res == REF)
