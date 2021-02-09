"""
This module implements the UnitaryMatrix class.

This is a concrete unitary matrix that can be operated on.
"""
from __future__ import annotations
from typing import Union

import numpy as np


class UnitaryMatrix():
    """The UnitaryMatrix Class."""

    def __init__(self, utry: np.ndarray): # TODO: Make utry take ndarraylike
        """
        Unitary Matrix Constructor
        """
        # TODO: Check is_unitary
        self.utry = utry
    
    def is_qubit_unitary(self) -> bool:
        """Returns true if this unitary can represent a qubit system."""
        return self.utry.shape[0] & (self.utry.shape[0] - 1) == 0
    
    def get_num_qubits(self) -> int:
        """Returns the number of qubits this unitary can represent."""
        if not self.is_qubit_unitary():
            raise TypeError("Unitary does not represent a pure qubit system.")
        return int(np.log2(len(self.utry)))
    
    def dagger(self) -> UnitaryMatrix:
        """Returns the conjugate transpose of the unitary matrix."""
        return UnitaryMatrix( self.utry.conj().T )
    
    @staticmethod
    def identity(dim: int) -> UnitaryMatrix:
        if dim <= 0:
            raise ValueError("Invalid dimension for identity matrix.")
        return UnitaryMatrix( np.identity(dim))

UnitaryLike: TypeAlias = Union[UnitaryMatrix, np.ndarray]
