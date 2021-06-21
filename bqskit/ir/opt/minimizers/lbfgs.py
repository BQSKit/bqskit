"""This module implements the LBFGSMinimizer class."""
from __future__ import annotations

import logging

from bqskitrs import LBFGSMinimizerNative

from bqskit.ir.opt.minimizer import Minimizer

_logger = logging.getLogger(__name__)


class LBFGSMinimizer(LBFGSMinimizerNative, Minimizer):
    """
    The LBFGSMinimizer class.

    The LBFGSMinimizer attempts to instantiate the circuit such that the
    circuit's cost, given by a CostFunction, is minimized.
    """
