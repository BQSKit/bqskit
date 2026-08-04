"""This module implements the YYGate."""
from __future__ import annotations

from openqudit.expressions import UnitaryExpression as _UnitaryExpression

from bqskit.ir.gate import Gate
from bqskit.utils.cachedclass import CachedClass


class YYGate(Gate, CachedClass):
    """
    The Ising YY coupling gate.

    The YY gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        \\frac{\\sqrt{2}}{2} & 0 & 0 & \\frac{\\sqrt{2}}{2}i \\\\
        0 & \\frac{\\sqrt{2}}{2} & -\\frac{\\sqrt{2}}{2}i & 0 \\\\
        0 & -\\frac{\\sqrt{2}}{2}i & \\frac{\\sqrt{2}}{2} & 0 \\\\
        \\frac{\\sqrt{2}}{2}i & 0 & 0 & \\frac{\\sqrt{2}}{2} \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _qasm_name = 'ryy(pi/2)'
    _expr = _UnitaryExpression(
        'YY() { [[sqrt(2)/2,0,0,i*sqrt(2)/2],'
        '[0,sqrt(2)/2,~i*sqrt(2)/2,0],'
        '[0,~i*sqrt(2)/2,sqrt(2)/2,0],'
        '[i*sqrt(2)/2,0,0,sqrt(2)/2]] }',
    )
