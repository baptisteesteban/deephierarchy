import numpy as np

from deephierarchy.graph import RAG

from heapq import heappush, heappop

def _bpt_core(rag_adjacency_matrix: np.ndarray, rag_weight_matrix: np.ndarray, _area: np.ndarray, _sum: np.ndarray) -> np.ndarray:
    ON = rag_adjacency_matrix.shape[0]
    N = 2 * ON - 1
    heap = []
    active_nodes = np.ones((N,), dtype=np.bool_)

    # This matrix represents the weighted adjacency matrix following the evolution of the graph during the computation.
    weights = np.full((N, N), -1, dtype=rag_weight_matrix.dtype)
    weights[:ON, :ON][rag_adjacency_matrix] = rag_weight_matrix[rag_adjacency_matrix]

    # We insert all the edges into the heap
    for n1 in range(ON):
        for n2 in range(n1 + 1, ON):
            if weights[n1, n2] >= 0:
                heappush(heap, (weights[n1, n2], (n1, n2)))

    # Attributes
    area = np.empty((N,), dtype=_area.dtype)
    area[:ON] = _area
    sum_shape = (N,) if _sum.ndim == 1 else (N, _sum.shape[-1])
    sum_v = np.empty(sum_shape, dtype=_sum.dtype)
    sum_v[:ON] = _sum

    # Main part of the algorithm
    parent = np.arange(N, dtype=np.uint32)
    num_nodes = ON
    while len(heap) > 0:
        w, (n1, n2) = heappop(heap)
        if not active_nodes[n1] or not active_nodes[n2]:
            continue

        # Create new BPT node and remove RAG node
        parent[n1] = num_nodes
        parent[n2] = num_nodes
        area[num_nodes] = area[n1] + area[n2]
        sum_v[num_nodes] = sum_v[n1] + sum_v[n2]
        active_nodes[n1] = active_nodes[n2] = False

        for nn in range(num_nodes):
            union_area = area[num_nodes] + area[nn]
            union_sum = sum_v[num_nodes] + area[nn]
            union_mean = union_sum / union_area
            dis = area[nn] * np.linalg.norm((sum_v[nn] / area[nn]) - union_mean) + area[num_nodes] * np.linalg.norm((sum_v[num_nodes] / area[num_nodes]) - union_mean)
            if weights[n1, nn] >= 0 or weights[n2, nn]:
                weight[num_nodes, nn] = weight[nn, num_nodes] = dis
                heappush(heap, (weight[num_nodes, nn], (num_nodes, nn)))

        num_nodes += 1
    
    return parent

def bpt(rag: RAG, edge_weights: np.ndarray, area: np.ndarray, sum_v: np.ndarray) -> np.ndarray:
    return _bpt_core(rag.adjacency_matrix, edge_weights, area, sum_v)
