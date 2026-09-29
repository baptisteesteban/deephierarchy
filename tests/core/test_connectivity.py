import numpy as np

from deephierarchy.core import C4, C8


def test_c4():
    REF = np.array([[3, 6], [2, 7], [3, 8], [4, 7]])
    p = np.array([3, 7])
    res = C4(p)

    assert C4.ndim == 2

    for ref, n in zip(REF, res):
        assert np.all(ref == n)


def test_c8():
    REF = np.array([[3, 6], [2, 6], [2, 7], [2, 8], [3, 8], [4, 8], [4, 7], [4, 6]])
    p = np.array([3, 7])
    res = C8(p)

    assert C8.ndim == 2

    for ref, n in zip(REF, res):
        assert np.all(ref == n)
