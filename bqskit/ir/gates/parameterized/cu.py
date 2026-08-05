"""This module implements the CUGate."""
from __future__ import annotations

import numpy as np
from openqudit.expressions import UnitaryExpression as _UnitaryExpression

from bqskit.ir.gate import Gate
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class CUGate(
    Gate,
    CachedClass,
):
    """
    A gate representing an arbitrary controlled rotation.

    It is given by the following parameterized unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & \\exp{i\\gamma}cos(\\frac{\\theta}{2})
        & -\\exp{i(\\gamma + \\lambda)}sin(\\frac{\\theta}{2}) \\\\
        0 & 0 & \\exp{i(\\gamma + \\phi)}sin(\\frac{\\theta}{2})
        & \\exp{i(\\gamma + \\phi + \\lambda)}cos(\\frac{\\theta}{2}) \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _num_params = 4
    _qasm_name = 'cu'
    _expr = _UnitaryExpression(
        'CU(t0,t1,t2,t3) { [[1,0,0,0],[0,1,0,0],'
        '[0,0,e^(i*t3)*cos(t0/2),~e^(i*(t3+t2))*sin(t0/2)],'
        '[0,0,e^(i*(t3+t1))*sin(t0/2),e^(i*(t3+t1+t2))*cos(t0/2)]] }',
    )

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        ct = np.cos(params[0] / 2)
        st = np.sin(params[0] / 2)
        cp = np.cos(params[1])
        sp = np.sin(params[1])
        cl = np.cos(params[2])
        sl = np.sin(params[2])
        cg = np.cos(params[3])
        sg = np.sin(params[3])
        el = cl + 1j * sl
        ep = cp + 1j * sp
        eg = cg + 1j * sg

        return UnitaryMatrix(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, eg * ct, -eg * el * st],
                [0, 0, eg * ep * st, eg * ep * el * ct],
            ],
        )
