"""This module implements the U2Gate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class U2Gate(QubitGate, DifferentiableUnitary, CachedClass):
    """
    The U2 single qubit gate.

    It is given by the following parameterized unitary:

    .. math::

        \\begin{pmatrix}
        \\frac{\\sqrt{2}}{2} & -\\exp({i\\theta_1})\\frac{\\sqrt{2}}{2} \\\\
        \\exp({i\\theta_0})\\frac{\\sqrt{2}}{2} &
         \\exp({i(\\theta_0 + \\theta_1)})\\frac{\\sqrt{2}}{2} \\\\
        \\end{pmatrix}
    """

    _num_qudits = 1
    _num_params = 2
    _qasm_name = 'u2'

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        sq2 = np.sqrt(2) / 2
        eip = np.exp(1j * params[0])
        eil = np.exp(1j * params[1])

        return UnitaryMatrix(
            [
                [sq2, -eil * sq2],
                [eip * sq2, eip * eil * sq2],
            ],
        )

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)

        sq2 = np.sqrt(2) / 2
        eip = np.exp(1j * params[0])
        eil = np.exp(1j * params[1])
        deip = 1j * np.exp(1j * params[0])
        deil = 1j * np.exp(1j * params[1])

        return np.array(
            [
                [  # wrt params[0]
                    [0, 0],
                    [deip * sq2, deip * eil * sq2],
                ],

                [  # wrt params[1]
                    [0, -deil * sq2],
                    [0, eip * deil * sq2],
                ],
            ], dtype=np.complex128,
        )
