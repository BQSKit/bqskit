"""This module implements the CSUMDGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.quditgate import QuditGate
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class CSUMGate(QuditGate):
    r"""
    The two-qudit Conditional-SUM (CSUM) gate.

    The CSUM_d gate is given by the following formula:

    .. math::
        \\begin{equation}
            CSUM |i,j> = |i, i + j mod d>
        \\end{equation}

    where d is the number of levels (2 levels is a qubit,
    3 levels is a qutrit, etc.)

    __init__() arguments:
        num_levels : int
            Number of levels in each qudit (d).
    """

    _num_qudits = 2
    _num_params = 0

    def __init__(self, num_levels: int=3):
        self.num_levels = num_levels

    def get_unitary(self) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""

        ival = 0
        jval = 0
        matrix = np.zeros([self.num_levels**2, self.num_levels**2])
        for i, col in enumerate(matrix.T):
            col[self.num_levels * jval + ((ival + jval) % self.num_levels)] = 1
            matrix[:, i] = col
            if ival == self.num_levels - 1:
                ival = 0
                jval += 1
            else:
                ival += 1
        u_mat = UnitaryMatrix(matrix, self.radixes)
        return u_mat
    
    def get_grad(self) -> npt.NDArray[np.complex128]:
        return np.array([])
