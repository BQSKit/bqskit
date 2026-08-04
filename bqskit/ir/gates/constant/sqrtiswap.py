"""This module implements the SqrtISwapGate."""
from __future__ import annotations

from openqudit.expressions import UnitaryExpression as _UnitaryExpression

from bqskit.ir.gate import Gate
from bqskit.utils.cachedclass import CachedClass


class SqrtISwapGate(Gate, CachedClass):
    """
    The square root two qubit swap and phase iSWAP gate.

    The SqrtISwap gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & \\frac{1}{\\sqrt{2}} & \\frac{i}{\\sqrt{2}} & 0 \\\\
        0 & \\frac{i}{\\sqrt{2}} & \\frac{1}{\\sqrt{2}} & 0 \\\\
        0 & 0 & 0 & 1 \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _qasm_name = 'sqisw'
    _expr = _UnitaryExpression(
        'SqrtISwap() { [[1,0,0,0],[0,1/sqrt(2),i/sqrt(2),0],'
        '[0,i/sqrt(2),1/sqrt(2),0],[0,0,0,1]] }',
    )
