"""This module implements the CeresMinimizer class."""
from __future__ import annotations

import logging
from typing import TypeVar

from bqskitrs import LeastSquaresMinimizerNative

from bqskit.ir.opt.minimizer import Minimizer

_logger = logging.getLogger(__name__)

# TODO: use PEP 673 Self type when mypy is upgraded
Self = TypeVar('Self', bound='CeresMinimizer')


class CeresMinimizer(LeastSquaresMinimizerNative, Minimizer):
    """
    The CeresMinimizer class.

    The CeresMinimizer attempts to instantiate the circuit such that the
    circuit's cost, given by a CostFunction, is minimized.
    """

    def __new__(
        cls: type[Self], num_threads: int = 1, ftol: float = 1e-6,
        gtol: float = 1e-10, report: bool = False,
    ) -> Self:
        """
        Create a new CeresMinimizer.

        Args:
            num_threads (int): The number of threads Ceres should use
                for optimization (defaults to 1)

            ftol (float): Default 1e-6. The function tolerance stopping
                condition Ceres should use, see the documentation about
                ftol here:
                http://ceres-solver.org/nnls_solving.html#_CPPv4N5ceres6Solver7Options18function_toleranceE

            gtol (float): Default 1e-10. The gradient tolerance stopping
                condition Ceres should use, see the documentation about
                gtol here:
                http://ceres-solver.org/nnls_solving.html#_CPPv4N5ceres6Solver7Options18gradient_toleranceE

            report (bool): Whether to print the minimization summary Ceres
                can generate. This is quite noisy so only enable this if
                you are certain you want it! Defaults to false.
        """

        return super().__new__(cls, num_threads, ftol, gtol, report)
