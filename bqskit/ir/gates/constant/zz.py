"""This module implements the ZZGate."""
from __future__ import annotations

import math

from openqudit.expressions import UnitaryExpression as _UnitaryExpression

from bqskit.ir.gate import Gate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class ZZGate(Gate, CachedClass):
    """
    The Ising ZZ coupling gate.

    The ZZ gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        \\frac{\\sqrt{2}}{2} - \\frac{\\sqrt{2}}{2}i & 0 & 0 & 0 \\\\
        0 & \\frac{\\sqrt{2}}{2} + \\frac{\\sqrt{2}}{2}i & 0 & 0 \\\\
        0 & 0 & \\frac{\\sqrt{2}}{2} + \\frac{\\sqrt{2}}{2}i & 0 \\\\
        0 & 0 & 0 & \\frac{\\sqrt{2}}{2} - \\frac{\\sqrt{2}}{2}i \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _qasm_name = 'rzz(pi/2)'
    _expr = _UnitaryExpression(
        'ZZ() { [[sqrt(2)/2-i*sqrt(2)/2,0,0,0],'
        '[0,sqrt(2)/2+i*sqrt(2)/2,0,0],'
        '[0,0,sqrt(2)/2+i*sqrt(2)/2,0],'
        '[0,0,0,sqrt(2)/2-i*sqrt(2)/2]] }',
    )
    _utry = UnitaryMatrix(
        [
            [math.sqrt(2) / 2 - 1j * math.sqrt(2) / 2, 0, 0, 0],
            [0, math.sqrt(2) / 2 + 1j * math.sqrt(2) / 2, 0, 0],
            [0, 0, math.sqrt(2) / 2 + 1j * math.sqrt(2) / 2, 0],
            [0, 0, 0, math.sqrt(2) / 2 - 1j * math.sqrt(2) / 2],
        ],
    )
