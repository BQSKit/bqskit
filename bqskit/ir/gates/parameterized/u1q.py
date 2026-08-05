"""This module implements the U1qGate."""

from __future__ import annotations

import math

import numpy as np
from openqudit.expressions import UnitaryExpression

from bqskit.ir.gate import Gate
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class U1qGate(Gate, CachedClass):
    """
    The Quantinuum U1q single qubit gate.

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
    _expr = UnitaryExpression(
        'U1q(θ0, θ1) {'
        '['
        '[cos(θ0/2), ~i*e^(~i*θ1)*sin(θ0/2)],'
        '[~i*e^(i*θ1)*sin(θ0/2), cos(θ0/2)]'
        ']'
        '}',
    )

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


U1qPiGate = U1qGate().with_frozen_params({0: math.pi})
U1qPi2Gate = U1qGate().with_frozen_params({0: math.pi / 2})
