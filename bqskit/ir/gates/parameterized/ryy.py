"""This module implements the RYYGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class RYYGate(
    QubitGate,
    DifferentiableUnitary,
    CachedClass,
):
    """
    A gate representing an arbitrary rotation around the YY axis.

    It is given by the following parameterized unitary:

    .. math::

        \\begin{pmatrix}
        \\cos{\\frac{\\theta}{2}} & 0 & 0 & \\sin{\\frac{\\theta}{2}}i \\\\
        0 & \\cos{\\frac{\\theta}{2}} & -\\sin{\\frac{\\theta}{2}}i & 0 \\\\
        0 & -\\sin{\\frac{\\theta}{2}}i & \\cos{\\frac{\\theta}{2}} & 0 \\\\
        \\sin{\\frac{\\theta}{2}}i & 0 & 0 & \\cos{\\frac{\\theta}{2}} \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'ryy'

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        cos = np.cos(params[0] / 2)
        nsin = -1j * np.sin(params[0] / 2)
        psin = 1j * np.sin(params[0] / 2)

        return UnitaryMatrix(
            [
                [cos, 0, 0, psin],
                [0, cos, nsin, 0],
                [0, nsin, cos, 0],
                [psin, 0, 0, cos],
            ],
        )

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)

        dcos = -np.sin(params[0] / 2) / 2
        dnsin = -1j * np.cos(params[0] / 2) / 2
        dpsin = 1j * np.cos(params[0] / 2) / 2

        return np.array(
            [
                [
                    [dcos, 0, 0, dpsin],
                    [0, dcos, dnsin, 0],
                    [0, dnsin, dcos, 0],
                    [dpsin, 0, 0, dcos],
                ],
            ], dtype=np.complex128,
        )
