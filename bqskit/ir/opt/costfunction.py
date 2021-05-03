"""This module implements the CostFunctionGenerator and CostFunction base classes."""

import abc
from typing import Sequence
import numpy as np

class CostFunction(abc.ABC):
    """
    The CostFunction base class.

    A CostFunction is a differentiable map from a vector of real numbers to 
    a real number.
    """

    @abc.abstractmethod
    def get_cost(self, params: Sequence[float]) -> float:
        """Return the cost value given the input parameters."""

    @abc.abstractmethod
    def get_grad(self, params: Sequence[float]) -> np.ndarray:
        """Return the cost gradient given the input parameters."""

    def get_cost_and_grad(
        self,
        params: Sequence[float]
    ) -> tuple[float, np.ndarray]:
        """Return the cost and gradient given the input parameters."""
        return self.get_cost(params), self.get_grad(params)