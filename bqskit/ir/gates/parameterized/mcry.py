"""This module implements the MCRYGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class MCRYGate(
    QubitGate,
    DifferentiableUnitary,
    CachedClass,
):
    """
    A gate representing a multi-controlled Y rotation.

    It is given by the following parameterized unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & \\cos{\\frac{\\theta}{2}} & -\\sin{\\frac{\\theta}{2}} \\\\
        0 & 0 & \\sin{\\frac{\\theta}{2}} & \\cos{\\frac{\\theta}{2}} \\\\
        \\end{pmatrix}
    """
    _qasm_name = 'mcry'

    def __init__(self, num_qudits: int) -> None:
        self._num_qudits = num_qudits
        self._num_params = 1
        super().__init__()

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        cos = np.cos(params[0] / 2)
        sin = np.sin(params[0] / 2)

        matrix = np.identity(2 ** self.num_qudits, dtype=np.complex128)
        matrix[-1, -1] = cos
        matrix[-2, -2] = cos
        matrix[-1, -2] = sin
        matrix[-2, -1] = -1 * sin

        return UnitaryMatrix(matrix)

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)

        dcos = -np.sin(params[0] / 2) / 2
        dsin = -1j * np.cos(params[0] / 2) / 2

        matrix = np.identity(2 ** self.num_qudits, dtype=np.complex128)
        matrix[-1, -1] = dcos
        matrix[-2, -2] = dcos
        matrix[-1, -2] = dsin
        matrix[-2, -1] = -1 * dsin

        return matrix
