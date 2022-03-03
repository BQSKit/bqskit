"""This module implements the CRYGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class CRYGate(
    QubitGate,
    DifferentiableUnitary,
    CachedClass,
):
    """
    A gate representing a controlled Y rotation.

    It is given by the following parameterized unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & \\cos{\\frac{\\theta}{2}} & -\\sin{\\frac{\\theta}{2}} \\\\
        0 & 0 & \\sin{\\frac{\\theta}{2}} & \\cos{\\frac{\\theta}{2}} \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'cry'

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        cos = np.cos(params[0] / 2)
        sin = np.sin(params[0] / 2)

        return UnitaryMatrix(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, cos, -sin],
                [0, 0, sin, cos],
            ],
        )

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)

        dcos = -np.sin(params[0] / 2) / 2
        dsin = -1j * np.cos(params[0] / 2) / 2

        return np.array(
            [
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, dcos, -dsin],
                    [0, 0, dsin, dcos],
                ],
            ], dtype=np.complex128,
        )
