"""This module implements the CeresMinimizer class."""
from __future__ import annotations

import logging

from bqskitrs import LeastSquaresMinimizerNative

from bqskit.ir.opt.minimizer import Minimizer

_logger = logging.getLogger(__name__)


class CeresMinimizer(LeastSquaresMinimizerNative, Minimizer):
    """
    The CeresMinimizer class.

    The CeresMinimizer attempts to instantiate the circuit such that the
    circuit's cost, given by a CostFunction, is minimized.
    """
