"""This module implements the LBFGSMinimizer class."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import scipy.optimize as opt

from bqskit.ir.opt.cost.differentiable import DifferentiableCostFunction
from bqskit.ir.opt.cost.function import CostFunction
from bqskit.ir.opt.minimizer import Minimizer

_logger = logging.getLogger(__name__)


class LBFGSMinimizer(Minimizer):
    """
    The LBFGSMinimizer class.

    The LBFGSMinimizer attempts to instantiate the circuit such that the
    circuit's cost, given by a CostFunction, is minimized.
    """

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        """Configure the minimizer."""
        self.options = kwargs

    def minimize(self, cost: CostFunction, x0: np.ndarray) -> np.ndarray:
        """Minimize the circuit with respect to some cost function."""

        if not isinstance(cost, DifferentiableCostFunction):
            raise RuntimeError(
                'L-BFGS optimizer requires a differentiable cost function.',
            )

        res = opt.minimize(
            cost.get_cost_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options=self.options,
        )

        if not res.success:
            _logger.warning(res.message)

        return res.x
