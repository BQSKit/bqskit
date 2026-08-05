"""This module implements the CRZGate."""

from __future__ import annotations

import numpy as np
from openqudit.expressions import UnitaryExpression as _UnitaryExpression

from bqskit.ir.gate import Gate
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class CRZGate(
    Gate,
    CachedClass,
):
    """
    A gate representing a controlled Z rotation.

    It is given by the following parameterized unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & \\exp({-i\\frac{\\theta}{2}}) & 0 \\\\
        0 & 0 & 0 & \\exp({i\\frac{\\theta}{2}}) \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'crz'
    _expr = _UnitaryExpression(
        'CRZ(t0) { [[1,0,0,0],[0,1,0,0],'
        '[0,0,e^(~i*t0/2),0],[0,0,0,e^(i*t0/2)]] }',
    )

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        pos = np.exp(1j * params[0] / 2)
        neg = np.exp(-1j * params[0] / 2)

        return UnitaryMatrix(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, neg, 0],
                [0, 0, 0, pos],
            ],
        )
