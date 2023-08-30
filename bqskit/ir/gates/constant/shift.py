"""This module implements the ShiftGate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates.quditgate import QuditGate
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class ShiftGate(QuditGate):
    r"""
    The one-qudit shift (X) gate. This is a Weyl-Heisenberg gate.

    The shift gate is given by the following formula:

    X = \sum_a |a + 1 mod d ><a|

    where d is the number of levels (2 levels is a qubit,
    3 levels is a qutrit, etc.)

    __init__() arguments:
        num_levels : int
            Number of levels in each qudit (d).
    """

    _num_qudits = 1
    _num_params = 0

    def __init__(self, num_levels: int):
        self.num_levels = num_levels

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""

        matrix = np.zeros([self.num_levels, self.num_levels], dtype=complex)
        for i, col in enumerate(matrix.T):
            if i + 1 >= self.num_levels:
                col[0] = 1
            else:
                col[i + 1] = 1
            matrix[:, i] = col
        u_mat = UnitaryMatrix(matrix, self.radixes)
        return u_mat
