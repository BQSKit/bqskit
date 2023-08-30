"""This module implements the PDGate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates.quditgate import QuditGate
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class PDGate(QuditGate):
    """
    The one-qudit P[i] gate.

    The clock gate is given by the following formula,
    taken from https://bit.ly/3MM6jou

    .. math::
        \\begin{equation}
            P[i]_d = \\sum_{j} (-\\omega**2)^{\\delta_{ij}} |j><j|
        \\end{equation}

    where
    .. math:: \\omega = \\exp(2\\pi*i/d)
    and d is the number of levels (2 levels is a qubit,
    3 levels is a qutrit, etc.)

    __init__() arguments:
        num_levels : int
            Number of levels in each qudit (d).
        ind: int
            The level on which to apply rotation (see above equation).
    """

    _num_qudits = 1
    _num_params = 0

    def __init__(self, num_levels: int, ind: int):
        self.num_levels = num_levels
        if ind > num_levels:
            raise ValueError(
                'PDGate index must be equal or less to the number of levels.',
            )
        self.ind = ind

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""

        diags = np.zeros(self.num_levels, dtype=complex)
        omega = np.exp(2j * np.pi / self.num_levels)
        for i in range(self.num_levels):
            if i == self.ind:
                diags[i] = -omega**2
            else:
                diags[i] = 1
        u_mat = UnitaryMatrix(np.diag(diags), self.radixes)
        return u_mat
