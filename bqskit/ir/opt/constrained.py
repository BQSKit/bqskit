"""This module implements the Minimizer base class."""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from bqskit.qis.unitary.unitary import RealVector

if TYPE_CHECKING:
    from bqskit.ir.opt.cost.function import CostFunction


class ConstrainedMinimizer(abc.ABC):
    """
    The Minimizer class.

    An minimizer finds the parameters for a circuit template that minimizes some
    CostFunction.
    """

    @abc.abstractmethod
    def constrained_minimize(
        self,
        cost: CostFunction,
        cstr: CostFunction,
        x0: RealVector,
    ) -> npt.NDArray[np.float64]:
        """
        Minimize `cost` starting from initial point `x0` while obeying `cstr`.

        Args:
            cost (CostFunction): The CostFunction to minimize.

            cstr (CostFunction): The CostFunction used to constrain solutions.

            x0 (np.ndarray): The initial point.

        Returns:
            (np.ndarray): The inputs that best minimizes the cost while obeying
                the constraints.

        Notes:
            This function should be side-effect free. This is because many
            calls may be running in parallel.
        """
