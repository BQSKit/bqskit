"""This module implements the ScipyMinimizer class."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

from bqskit.ir.opt.minimizer import Minimizer

if TYPE_CHECKING:
    from bqskit.ir.opt.cost import CostFunction
    from bqskit.qis import RealVector
    import numpy.typing as npt


class ScipyMinimizer(Minimizer):
    """A minimizer that uses scipy's 'minimize' function."""

    def __init__(self, tol: float = 1e-10) -> None:
        """
        Construct a ScipyMinimizer by passing all arguments to scipy.

        Args:
            tol (float): Set the tolerance used during scipy's minimize.
        """
        self.tol = tol

    def minimize(
        self,
        cost: CostFunction,
        x0: RealVector,
    ) -> npt.NDArray[np.float64]:
        """Solve the minimization problem by using scipy."""
        if len(x0) == 0:
            return np.array([])
        return minimize(cost, x0, tol=self.tol).x
