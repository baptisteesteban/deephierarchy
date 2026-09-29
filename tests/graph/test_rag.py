import numpy as np

from deephierarchy.core import C4, C8
from deephierarchy.graph import build_rag

def test_rag_c4():
    labels = np.array([
        [0, 0, 1],
        [1, 1, 2],
        [3, 3, 4]
    ], dtype=int)
    REF = np.array([
        [False, True, False, False, False],
        [True, False, True, True, False],
        [False, True, False, False, True],
        [False, True, False, False, True],
        [False, False, True, True, False]
    ])
    rag = build_rag(labels, C4)
    assert rag.num_nodes == 5
    assert np.all(rag.adjacency_matrix == REF)

def test_rag_c8():
    labels = np.array([
        [0, 0, 1],
        [1, 1, 2],
        [3, 3, 4]
    ], dtype=int)
    REF = np.array([
        [False, True, True, False, False],
        [True, False, True, True, True],
        [True, True, False, True, True],
        [False, True, True, False, True],
        [False, True, True, True, False]
    ])
    rag = build_rag(labels, C8)
    assert rag.num_nodes == 5
    assert np.all(rag.adjacency_matrix == REF)