"""This module implements the MCRZGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class MCRZGate(
    QubitGate,
    DifferentiableUnitary,
    CachedClass,
):
    """
    A gate representing a multi-controlled Z rotation.

    It is given by the following parameterized unitary:
    """
    _qasm_name = 'mcrz'

    def __init__(self, num_qudits: int) -> None:
        self._num_qudits = num_qudits
        self._num_params = 1
        super().__init__()

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        pos = np.exp(1j * params[0] / 2)
        neg = np.exp(-1j * params[0] / 2)

        matrix = np.identity(2 ** self.num_qudits, dtype=np.complex128)
        matrix[-1, -1] = pos
        matrix[-2, -2] = neg

        return UnitaryMatrix(matrix)

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)

        dpos = 1j / 2 * np.exp(1j * params[0] / 2)
        dneg = -1j / 2 * np.exp(-1j * params[0] / 2)

        matrix = np.identity(2 ** self.num_qudits, dtype=np.complex128)
        matrix[-1, -1] = dpos
        matrix[-2, -2] = dneg

        return matrix
