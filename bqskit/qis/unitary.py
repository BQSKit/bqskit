"""
This module implements the Unitary base class.

Represents a unitary matrix that can be retrieved from get_unitary.
"""
import abc
from typing import List
from typing import Optional

import numpy as np


class Unitary (abc.ABC):
    """Unitary Base Class."""

    @abc.abstractmethod
    def get_unitary(self, params: Optional[List[float]] = None) -> np.ndarray:
        """
        Abstract method that should return this unitary
        as a numpy matrix.

        Returns:
            (np.ndarray): The unitary matrix.
        """
