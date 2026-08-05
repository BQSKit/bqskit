"""This module implements the CCPGate."""
from __future__ import annotations

import numpy as np
from openqudit.expressions import UnitaryExpression as _UnitaryExpression

from bqskit.ir.gate import Gate
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class CCPGate(
    Gate,
    CachedClass,
):
    """
    A gate representing a controlled controlled phase rotation.

    It is given by the following parameterized unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\\\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\\\
        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\\\
        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & \\exp({i\\theta}) \\\\
        \\end{pmatrix}
    """

    _num_qudits = 3
    _num_params = 1
    _qasm_name = 'ccp'
    _expr = _UnitaryExpression(
        'CCP(t0) { ['
        '[1,0,0,0,0,0,0,0],'
        '[0,1,0,0,0,0,0,0],'
        '[0,0,1,0,0,0,0,0],'
        '[0,0,0,1,0,0,0,0],'
        '[0,0,0,0,1,0,0,0],'
        '[0,0,0,0,0,1,0,0],'
        '[0,0,0,0,0,0,1,0],'
        '[0,0,0,0,0,0,0,e^(i*t0)]] }',
    )

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        exp = np.exp(1j * params[0])

        return UnitaryMatrix(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, exp],
            ],
        )
