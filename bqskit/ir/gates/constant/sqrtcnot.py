"""This module implements the SqrtCNOTGate."""
from __future__ import annotations

from openqudit.expressions import UnitaryExpression as _UnitaryExpression

from bqskit.ir.gate import Gate
from bqskit.utils.cachedclass import CachedClass


class SqrtCNOTGate(Gate, CachedClass):
    """
    The Square root Controlled-X gate.

    The SqrtCNOT gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & \\frac{1}{2} + \\frac{1}{2}i & \\frac{1}{2} - \\frac{1}{2}i \\\\
        0 & 0 & \\frac{1}{2} - \\frac{1}{2}i & \\frac{1}{2} + \\frac{1}{2}i \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _qasm_name = 'csx'
    _expr = _UnitaryExpression(
        'SqrtCNOT() { [[1,0,0,0],[0,1,0,0],'
        '[0,0,0.5+0.5*i,0.5-0.5*i],[0,0,0.5-0.5*i,0.5+0.5*i]] }',
    )
