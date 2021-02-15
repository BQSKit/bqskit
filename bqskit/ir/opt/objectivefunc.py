from __future__ import annotations

import abc
from typing import Sequence

import numpy as np

from bqskit.qis.unitarymatrix import UnitaryMatrix


class ObjectiveFunction(abc.ABC):
    """An ObjectiveFunction in BQSKit is a real-valued function that maps
    unitary matrices to real numbers."""

    @abc.abstractmethod
    def cost(self, U: UnitaryMatrix) -> float:
        """Returns the cost given a circuit's unitary."""

    @abc.abstractmethod
    def cost_grad(self, dU: Sequence[np.ndarray]) -> list[float]:
        """Returns the gradient of the cost function, given the circuit's
        gradient."""
