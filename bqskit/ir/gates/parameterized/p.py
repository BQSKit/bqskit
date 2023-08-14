"""This module implements the PGate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates.quditgate import QuditGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass
from bqskit.utils.typing import is_integer


class PGate(QuditGate, DifferentiableUnitary, CachedClass):
    """
    A gate representing the phase gate for qudits.

    __init__() arguments:
        num_levels : int
            Number of levels in each qudit (d).
        level: int
             The level on which to apply the phase gate (0...d-1).
    get_unitary arguments:
            param: float
            The angle by which to rotate
    """

    _num_qudits = 1
    _num_params = 1

    def __init__(
        self,
        num_levels: int = 2, 
        level: int = 1
    ) -> None:
        """
            Args:
            num_levels (int): The number of qudit levels (>=2).

            level (int): the level for the  phase qudit gate (<num_levels)
            
            Raises:
            ValueError: if num_levels < 2
            ValueError: if level >= num_levels
        """
        if num_levels < 2 or not is_integer(num_levels):
            raise ValueError(
                'PGate num_levels must be a postive integer greater than or equal to 2.',
            )
        self.num_levels = num_levels
        if level > num_levels:
            raise ValueError(
                'PGate index must be equal or less to the number of levels.',
            )
        self.level = level

    def get_unitary(
        self, 
        params: RealVector = []
    ) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        matrix = np.eye(self.num_levels, dtype=complex)
        matrix[self.level, self.level] = np.exp(params[0] * 1j)
        return UnitaryMatrix(matrix, self.radixes)

    def get_grad(
        self, 
        params: RealVector = []
    ) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)

        matrix = np.eye(self.num_levels, dtype=complex)
        matrix[self.level, self.level] = np.exp(params[0] * 1j) * 1j

        return np.array(
            [
                matrix,
            ], dtype=np.complex128,
        )
