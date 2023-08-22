"""This module implements the RZZGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class RZZGate(
    QubitGate,
    DifferentiableUnitary,
    CachedClass,
):
    """
    A gate representing an arbitrary rotation around the ZZ axis.

    It is given by the following parameterized unitary:

    .. math::

        \\begin{pmatrix}
        \\exp({-i\\frac{\\theta}{2}}) & 0 & 0 & 0 \\\\
        0 & \\exp({i\\frac{\\theta}{2}}) & 0 & 0 \\\\
        0 & 0 & \\exp({i\\frac{\\theta}{2}}) & 0 \\\\
        0 & 0 & 0 & \\exp({-i\\frac{\\theta}{2}}) \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'rzz'

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        pos = np.exp(1j * params[0] / 2)
        neg = np.exp(-1j * params[0] / 2)

        return UnitaryMatrix(
            [
                [neg, 0, 0, 0],
                [0, pos, 0, 0],
                [0, 0, pos, 0],
                [0, 0, 0, neg],
            ],
        )

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)

        dpos = 1j / 2 * np.exp(1j * params[0] / 2)
        dneg = -1j / 2 * np.exp(-1j * params[0] / 2)

        return np.array(
            [
                [
                    [dneg, 0, 0, 0],
                    [0, dpos, 0, 0],
                    [0, 0, dpos, 0],
                    [0, 0, 0, dneg],
                ],
            ], dtype=np.complex128,
        )
