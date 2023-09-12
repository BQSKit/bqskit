"""This module implements the CUGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class CUNGate(
    QubitGate,
    CachedClass,
):
    """
    A gate representing an arbitrary controlled rotation on N gates.

    It is given by the following parameterized unitary:

    .. math::

        \\begin{pmatrix}
        1 &  0  &  0  & 0 \\\\
        0 &  1  &  0  & 0 \\\\
        0 &  0  & U...&U \\\\
        . & ... & ... & . \\\
        . &     &     & .
        . &     &     & .
        0 &  0  & U...& U \\\\
    """

    _qasm_name = 'cun'

    def __init__(self, num_qudits: int, num_control: int) -> None:
        self._num_qudits = num_qudits
        self.num_control = num_control

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        matrix = np.zeros((2 ** self._num_qudits, 2 ** self._num_qudits))
        self.check_parameters(params)
        mid = len(params) // 2
        real = np.array(params[:mid], dtype=np.complex128)
        imag = 1j * np.array(params[mid:], dtype=np.complex128)
        x = real + imag
        start = 2 ** self.num_control
        matrix[start:, start:] = x
        matrix[0:start, 0:start] = np.identity(start)

        return UnitaryMatrix(matrix)
