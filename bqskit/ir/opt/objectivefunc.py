from __future__ import annotations

import abc
from typing import Sequence

import numpy as np

from bqskit.ir.circuit import Circuit
from bqskit.qis.unitarymatrix import UnitaryMatrix


class ObjectiveFunction(abc.ABC):
    """An ObjectiveFunction in BQSKit is a real-valued function that maps
    unitary matrices to real numbers."""

    @abc.abstractmethod
    def cost(self, circuit: Circuit) -> float:
        """Returns the cost given a circuit."""

    @abc.abstractmethod
    def cost_grad(self, circuit: Circuit) -> list[float]:
        """Returns the gradient of the cost function."""
