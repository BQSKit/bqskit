"""This module implements the U1qGate."""
from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class U1qGate(QubitGate, DifferentiableUnitary, CachedClass):
    """
    The Honeywell U1q single qubit gate.

    It is given by the following parameterized unitary:

    .. math::

        \\begin{pmatrix}
        \\cos{\\frac{\\theta_0}{2}} &
         -i\\exp{-i\\theta_1}\\sin{\\frac{\\theta_0}{2}} \\\\
        -i\\exp{i\\theta_1}\\sin{\\frac{\\theta_0}{2}} &
         \\cos{\\frac{\\theta_0}{2}} \\\\
        \\end{pmatrix}
    """

    _num_qudits = 1
    _num_params = 2
    _qasm_name = 'U1q'

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        ct = np.cos(params[0] / 2)
        st = np.sin(params[0] / 2)
        enp = np.exp(-1j * params[1])
        epp = np.exp(1j * params[1])

        return UnitaryMatrix(
            [
                [ct, -1j * enp * st],
                [-1j * epp * st, ct],
            ],
        )

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)

        ct = np.cos(params[0] / 2)
        st = np.sin(params[0] / 2)
        enp = np.exp(-1j * params[1])
        epp = np.exp(1j * params[1])

        return np.array(
            [
                [  # wrt params[0]
                    [-0.5 * st, -0.5j * enp * ct],
                    [-0.5j * epp * ct, -0.5 * st],
                ],

                [  # wrt params[1]
                    [0, -1 * enp * st],
                    [epp * st, 0],
                ],
            ], dtype=np.complex128,
        )


U1qPiGate = U1qGate().with_frozen_params({0: math.pi})
U1qPi2Gate = U1qGate().with_frozen_params({0: math.pi / 2})
