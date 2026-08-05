"""This module implements the SqrtTGate."""

from __future__ import annotations

from openqudit.expressions import UnitaryExpression as _UnitaryExpression

from bqskit.ir.gate import Gate
from bqskit.utils.cachedclass import CachedClass


class SqrtTGate(Gate, CachedClass):
    """
    The single-qubit square root T gate.

    .. math::

        \\begin{pmatrix}
        1 & 0 \\\\
        0 & e^{i\\frac{\\pi}{8}} \\\\
        \\end{pmatrix}
    """

    _num_qudits = 1
    _qasm_name = 'st'
    _expr = _UnitaryExpression('SqrtT() { [[1,0],[0,e^(i*pi/8)]] }')
