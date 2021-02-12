"""
This module implements the Unitary abstract base class.

Represents a UnitaryMatrix that can be retrieved from get_unitary.
"""
from __future__ import annotations

import abc
from typing import Optional, Sequence

import numpy as np

from bqskit.qis.unitarymatrix import UnitaryMatrix


class Unitary (abc.ABC):
    """Unitary Base Class."""

    @abc.abstractmethod
    def get_unitary(self, params: Optional[Sequence[float]] = None) -> UnitaryMatrix:
        """
        Abstract method that should return this unitary as a UnitaryMatrix.

        Args:
            params (Optional[Sequence[float]]): Unconstrained real number
                parameters for parameterized unitaries.

        Returns:
            (UnitaryMatrix): The unitary matrix.
        """
