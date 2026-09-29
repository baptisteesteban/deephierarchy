import numpy as np

from typing import Iterable


class Connectivity:
    def __init__(self, data: Iterable[int]):
        assert data.ndim == 2
        self._data = np.asarray(data, dtype=int)

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def ndim(self) -> int:
        return self._data.shape[1]


C4 = Connectivity([[0, -1], [-1, 0], [0, 1], [1, 0]])

C8 = Connectivity(
    [[0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1]]
)
