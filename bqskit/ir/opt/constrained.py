"""This module implements the ConstrainedMinimizer base class."""
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
    The ConstrainedMinimizer class.

    A minimizer that finds parameters for a circuit template that minimize a
    'cost' CostFunction while also satisfying some 'constraint' CostFunction.
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
            cost (CostFunction): The CostFunction to minimize. This function
                should capture the objective of the optimization.

            cstr (CostFunction): The CostFunction used to constrain solutions.
                In most cases, this will be based on the Hilbert-Schmidt dist-
                ance or some related fidelity metric.

            x0 (np.ndarray): An initial point in parameter space.

        Returns:
            (np.ndarray): The inputs that minimize the cost while obeying the
                constraints.

        Notes:
            This function should be side-effect free. This is because many
            calls may be running in parallel.
        """
