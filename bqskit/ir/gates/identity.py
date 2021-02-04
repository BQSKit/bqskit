"""This module implements the Identity Gate."""
from __future__ import annotations


import numpy as np

from bqskit.ir.fixedgate import FixedGate
from bqskit.qis.unitarymatrix import UnitaryMatrix

class IdentityGate(FixedGate):
    """An N-qudit identity unitary."""

    def __init__(self, size: int = 1, radix: int = 2):
        """Create an IdentityGate, defaulting to a 1 qubit identity.

        Args:
            size (int): The number of qudits the identity should span
            radix (int): The radix of the identity (qubit vs qudit)

        """
        self.size = size
        self.radixes = [radix] * size
        self.utry = UnitaryMatrix(np.array(np.eye(radix**size), dtype=np.complex128))
