"""This module implements the LBFGSMinimizer class."""
from __future__ import annotations

from typing import Any

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit
from bqskit.ir.opt.costfunction import CostFunction
from bqskit.ir.opt.minimizer import Minimizer
import scipy.optimize as opt


class LBFGSMinimizer(Minimizer):
    """
    The LBFGSMinimizer class.

    The LBFGSMinimizer attempts to instantiate the circuit such that the
    circuit's cost, given by a CostFunction, is minimized.
    """

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        """Configure the minimizer."""
        self.options = kwargs

    def minimize(self, circuit: Circuit, cost: CostFunction) -> np.ndarray:
        """Minimize the circuit with respect to some cost function."""

        res = opt.minimize(
            cost.get_cost_and_grad,
            circuit.get_params(),
            jac=True,
            method='L-BFGS-B',
            options=self.options,
        )

        return res.x
