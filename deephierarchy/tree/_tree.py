import numpy as np

# TODO:
# * Cut from number of element
# * Cut from altitude
# * Saliency map


class Tree:
    def __init__(
        self, parent: np.ndarray, node_map: np.ndarray, altitude: np.ndarray | None = None
    ):
        self._parent = parent
        self._node_map = node_map
        self._altitude = altitude

    @property
    def parent(self) -> np.ndarray:
        return self._parent

    @property
    def node_map(self) -> np.ndarray:
        return self._node_map

    @property
    def altitude(self) -> np.ndarray | None:
        return self._altitude
