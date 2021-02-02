"""
This module implements the UnitaryMatrix class.

This is a concrete unitary matrix that can be operated on.
"""
from typing import List
from typing import Optional

import numpy as np

from bqskit.qis import Unitary


class UnitaryMatrix (Unitary):
    """The UnitaryMatrix Class."""

    def __init__(self, utry: np.ndarray):
        """
        Unitary Matrix Constructor
        """
        self.utry = utry

    def get_unitary(self, params: Optional[List[float]] = None) -> np.ndarray:
        return self.utry
