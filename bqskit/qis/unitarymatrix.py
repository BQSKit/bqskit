"""
This module implements the UnitaryMatrix class.

This is a concrete unitary matrix that can be operated on.
"""
from __future__ import annotations

import numpy as np


class UnitaryMatrix():
    """The UnitaryMatrix Class."""

    def __init__(self, utry: np.ndarray):
        """
        Unitary Matrix Constructor
        """
        self.utry = utry
