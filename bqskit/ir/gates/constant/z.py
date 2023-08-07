"""This module implements the ZGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.quditgate import QuditGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_integer


class ZGate(QuditGate):
    """
    The one-qudit Z[i,j] gate. This gate is equivalent to a Pauli Z gate on the subspace of levels i,j.
    """
    _num_qudits = 1
    _num_params = 0
    _qasm_name = 'z'

    def __init__(
        self, 
        num_levels: int = 2, 
        level_1: int = 0, 
        level_2: int = 1
    ) -> None:
        """
            Args:
            num_levels (int): The number of qudit levels (>=2).

            level_1 (int): the first level for the Z qudit gate (<num_levels)
            level_2 (int): the second level for the Z qudit gate (<num_levels)
            
            Raises:
            ValueError: if num_levels < 2
            ValueError: if any of levels >= num_levels
        """
        if num_levels < 2 or not is_integer(num_levels):
            raise ValueError(
                'ZGate num_levels must be a postive integer greater than or equal to 2.',
            )
        self.num_levels = num_levels
        if level_1 > num_levels or level_2 > num_levels:
            raise ValueError(
                'ZGate indices must be equal or less to the number of levels.',
            )
        self.level_1 = level_1
        self.level_2 = level_2


    def get_unitary(self) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""

        matrix = np.eye(self.num_levels, dtype=complex)
        matrix[self.level_2, self.level_2] = 1.0 
        matrix[self.level_2, self.level_2] = -1.0
        return UnitaryMatrix(matrix, self.radixes)

    def get_grad(self) -> npt.NDArray[np.complex128]:
        return np.array([])
