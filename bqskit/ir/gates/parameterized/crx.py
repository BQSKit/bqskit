"""This module implements the CRXGate."""
from __future__ import annotations

import numpy as np
from openqudit.expressions import UnitaryExpression as _UnitaryExpression

from bqskit.ir.gate import Gate
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class CRXGate(
    Gate,
    CachedClass,
):
    """
    A gate representing a controlled X rotation.

    It is given by the following parameterized unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & \\cos{\\frac{\\theta}{2}} & -\\sin{\\frac{\\theta}{2}}i \\\\
        0 & 0 & -\\sin{\\frac{\\theta}{2}}i & \\cos{\\frac{\\theta}{2}} \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'crx'
    _expr = _UnitaryExpression(
        'CRX(t0) { [[1,0,0,0],[0,1,0,0],'
        '[0,0,cos(t0/2),~i*sin(t0/2)],'
        '[0,0,~i*sin(t0/2),cos(t0/2)]] }',
    )

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        cos = np.cos(params[0] / 2)
        sin = -1j * np.sin(params[0] / 2)

        return UnitaryMatrix(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, cos, sin],
                [0, 0, sin, cos],
            ],
        )
