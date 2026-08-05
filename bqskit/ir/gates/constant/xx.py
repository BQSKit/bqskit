"""This module implements the XXGate."""

from __future__ import annotations

from openqudit.expressions import UnitaryExpression as _UnitaryExpression

from bqskit.ir.gate import Gate
from bqskit.utils.cachedclass import CachedClass


class XXGate(Gate, CachedClass):
    """
    The Ising XX coupling gate.

    The XX gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        \\frac{\\sqrt{2}}{2} & 0 & 0 & -\\frac{\\sqrt{2}}{2}i \\\\
        0 & \\frac{\\sqrt{2}}{2} & -\\frac{\\sqrt{2}}{2}i & 0 \\\\
        0 & -\\frac{\\sqrt{2}}{2}i & \\frac{\\sqrt{2}}{2} & 0 \\\\
        -\\frac{\\sqrt{2}}{2}i & 0 & 0 & \\frac{\\sqrt{2}}{2} \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _qasm_name = 'rxx(pi/2)'
    _expr = _UnitaryExpression(
        'XX() { [[sqrt(2)/2,0,0,~i*sqrt(2)/2],'
        '[0,sqrt(2)/2,~i*sqrt(2)/2,0],'
        '[0,~i*sqrt(2)/2,sqrt(2)/2,0],'
        '[~i*sqrt(2)/2,0,0,sqrt(2)/2]] }',
    )
