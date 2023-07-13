"""This module implements the HDGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.quditgate import QuditGate
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_integer


class HGate(QuditGate):
    """
    The one-qudit Hadamard gate. This is a Clifford gate.

    The clock gate is given by the following formula:

    .. math::
        \\begin{equation}
            H_d = 1/\\sqrt(d) \\sum_{ij} \\omega_d^{ij} |i >< j|
        \\end{equation}

    where
    .. math:: \\omega = \\exp(2\\pi*i/d)
    and d is the number of levels (2 levels is a qubit,
    3 levels is a qutrit, etc.)

    __init__() arguments:
        num_levels : int
            Number of levels in each quantum object.
    """

    _num_qudits = 1
    _num_params = 0
    _qasm_name = 'h'

    def __init__(self, num_levels: int):
        if num_levels < 2 or not is_integer(num_levels):
            raise ValueError(
                'HGate num_levels must be a postive integer greater than or equal to 2.',
            )
        self.num_levels = num_levels

    def get_unitary(self) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""

        matrix = np.zeros([self.num_levels, self.num_levels], dtype=complex)
        omega = np.exp(2j * np.pi / self.num_levels)
        for i in range(self.num_levels):
            for j in range(i, self.num_levels):
                matrix[i, j] = omega**(i * j)
                matrix[j, i] = omega**(i * j)
        u_mat = UnitaryMatrix(
            matrix * 1 / np.sqrt(self.num_levels), self.radixes,
        )
        return u_mat

    def get_grad(self) -> npt.NDArray[np.complex128]:
        return np.array([])
