import numpy as np

from deephierarchy.core import C4
from deephierarchy.graph import build_rag, weight_rag_weighted_area_dist
from deephierarchy.hierarchy import bpt
from deephierarchy.labeling import compute_area, compute_sum


def test_bpt_weighted_distance():
    label = np.array([[0, 0, 1, 2], [0, 0, 1, 1], [0, 3, 3, 3]], dtype=int)
    img = np.array([[1, 1, 3, 10], [2, 1, 4, 3], [1, 5, 6, 5]], dtype=np.uint8)
    REF = np.array([6, 4, 5, 4, 5, 6, 6], dtype=int)

    rag = build_rag(label, C4)
    sum_v = compute_sum(label, img)
    area = compute_area(label)
    weight = weight_rag_weighted_area_dist(rag, area, sum_v)
    t = bpt(rag, weight, area, sum_v, label)
    assert np.all(label == t.node_map)
    assert np.all(REF == t.parent)
    assert t.altitude is None
