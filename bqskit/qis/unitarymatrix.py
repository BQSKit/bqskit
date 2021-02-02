"""
This module implements the UnitaryMatrix class.

This is a concrete unitary matrix that can be operated on.
"""
from __future__ import annotations

import numpy as np

from bqskit.qis import Unitary


class UnitaryMatrix (Unitary):
    """The UnitaryMatrix Class."""

    def __init__(self, utry: np.ndarray):
        """
        Unitary Matrix Constructor
        """
        self.utry = utry

    def get_unitary(self, params: list[float] | None = None) -> np.ndarray:
        return self.utry
