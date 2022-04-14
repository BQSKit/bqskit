"""This module implements the CeresMinimizer class."""
from __future__ import annotations

import logging

from bqskitrs import LeastSquaresMinimizerNative
from typing_extensions import Self

from bqskit.ir.opt.minimizer import Minimizer

_logger = logging.getLogger(__name__)


class CeresMinimizer(LeastSquaresMinimizerNative, Minimizer):
    """
    The CeresMinimizer class.

    The CeresMinimizer attempts to instantiate the circuit such that the
    circuit's cost, given by a CostFunction, is minimized.
    """

    def __new__(
        cls: type[Self], num_threads: int = 1, ftol: float = 1e-6,  # type: ignore # Remove once updating to mypy 0.950
        gtol: float = 1e-10, report: bool = False,
    ) -> Self:  # type: ignore # Remove once updating to mypy 0.950
        return super().__new__(cls, num_threads, ftol, gtol, report)
