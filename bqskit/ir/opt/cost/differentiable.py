"""This module implements DifferentiableCostFunction base classes."""
from __future__ import annotations

import abc
from typing import Sequence

import numpy as np

from bqskit.ir.opt.cost.function import CostFunction


class DifferentiableCostFunction(CostFunction):
    """
    The DifferentiableCostFunction base class.

    A DifferentiableCostFunction is a differentiable map from a vector of
    real numbers to a real number.

    A DifferentiableCostFunction exposes the `get_grad` abstract method
    and the `get_cost_and_grad` method. When subclassing, you only need to
    implement the `gen_cost` and `gen_grad` function factories. You can
    overwrite gen_cost_and_grad for optimization in some cases.
    """

    @abc.abstractmethod
    def get_grad(self, params: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return the cost gradient given the input parameters."""

    def get_cost_and_grad(
        self,
        params: Sequence[float] | np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Return the cost and gradient given the input parameters."""
        return self.get_cost(params), self.get_grad(params)
