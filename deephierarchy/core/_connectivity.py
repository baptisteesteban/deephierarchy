from typing import Iterable

import numpy as np


class Connectivity:
    def __init__(self, data: Iterable[int]):
        offsets = np.asarray(data, dtype=int)
        assert offsets.ndim == 2
        self._data = np.asarray(data, dtype=int)

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def ndim(self) -> int:
        return self._data.shape[1]

    def __call__(self, p: np.ndarray) -> np.ndarray:
        assert p.shape[-1] == self.ndim
        return p + self.data


C4 = Connectivity([[0, -1], [-1, 0], [0, 1], [1, 0]])

C8 = Connectivity([[0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1]])
