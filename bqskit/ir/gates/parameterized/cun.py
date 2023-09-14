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
        self.num_control = int(num_control)
        self._num_qudits = int(num_qudits)
        self._dim = int(2 ** (num_qudits))
        self.shape = (self.dim, self.dim)
        u_dim = int(self._dim / (2 ** self.num_control))
        self.u_shape = (u_dim, u_dim)
        self._num_params = 2 * (4 ** (num_qudits - num_control))
        self._name = 'CUNGate(%d, %s)' % (
            self.num_qudits, str(self.num_control),
        )

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)
        matrix = np.identity(2 ** self._num_qudits, dtype=np.complex128)
        mid = len(params) // 2
        real = np.array(params[:mid], dtype=np.complex128)
        imag = 1j * np.array(params[mid:], dtype=np.complex128)
        x = real + imag
        # Only last block is the unitary
        start = self.dim - self.u_shape[0]
        x = np.reshape(x, self.u_shape)
        matrix[start:, start:] = x
        return UnitaryMatrix(matrix)
