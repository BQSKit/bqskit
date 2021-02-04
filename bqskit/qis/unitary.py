"""
This module implements the Unitary base class.

Represents a unitary matrix that can be retrieved from get_unitary.
"""
from __future__ import annotations

import abc

import numpy as np

from bqskit.qis.unitarymatrix import UnitaryMatrix


class Unitary (abc.ABC):
    """Unitary Base Class."""

    @abc.abstractmethod
    def get_unitary(self, params: Optional[list[float]] = None) -> UnitaryMatrix:
        """
        Abstract method that should return this unitary
        as a numpy matrix.

        Returns:
            (np.ndarray): The unitary matrix.
        """
