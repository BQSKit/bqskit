"""
This module implements the Unitary base class.

Represents a unitary matrix that can be retrieved from get_unitary.
"""

import abc


class Unitary ( abc.ABC ):
    """Unitary Base Class."""

    @abc.abstractmethod    
    def get_unitary ( self, params = None ):
        """
        Abstract method that should return this unitary
        as a numpy matrix.

        Returns:
            (np.ndarray): The unitary matrix.
        """

