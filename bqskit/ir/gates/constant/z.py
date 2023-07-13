"""This module implements the ZGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.quditgate import QuditGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_integer


class ZGate(QuditGate):
    """
    The one-qudit Z[i] gate. This gate is equivalent to a Pauli Z gate on the
    level i.

    __init__() arguments:
        num_levels : int
            Number of levels in each qudit (d).
        level: int
            The level on which to apply the z gate (0...d-1).
    """
    _num_qudits = 1
    _num_params = 0
    _qasm_name = 'z'

    def __init__(self, num_levels: int = 2, level: int = 1):
        if num_levels < 2 or not is_integer(num_levels):
            raise ValueError(
                'ZGate num_levels must be a postive integer greater than or equal to 2.',
            )
        self.num_levels = num_levels
        if level > num_levels:
            raise ValueError(
                'ZGate index must be equal or less to the number of levels.',
            )
        self.level = level

    def get_unitary(self) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""

        matrix = np.eye(self.num_levels, dtype=complex)
        matrix[self.level, self.level] = -1.0
        return UnitaryMatrix(matrix, self.radixes)

    def get_grad(self) -> npt.NDArray[np.complex128]:
        return np.array([])
