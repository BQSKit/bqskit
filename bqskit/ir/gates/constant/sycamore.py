"""This module implements the SycamoreGate."""

from __future__ import annotations

from openqudit.expressions import UnitaryExpression as _UnitaryExpression

from bqskit.ir.gate import Gate
from bqskit.utils.cachedclass import CachedClass


class SycamoreGate(Gate, CachedClass):
    """
    The SycamoreGate gate.

    The Sycamore gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 0 & -i & 0 \\\\
        0 & -i & 0 & 0 \\\\
        0 & 0 & 0 & e^{-i\\frac{\\pi}{6}} \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _qasm_name = 'syc'
    _expr = _UnitaryExpression(
        'Sycamore() { [[1,0,0,0],[0,0,~i,0],[0,~i,0,0],[0,0,0,e^(~i*pi/6)]] }',
    )
