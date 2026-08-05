"""This module implements the BGate."""

from __future__ import annotations

from openqudit.expressions import UnitaryExpression as _UnitaryExpression

from bqskit.ir.gate import Gate
from bqskit.utils import CachedClass


class BGate(Gate, CachedClass):
    """
    The 2 qubit B gate.

    The B gate is given by the following unitary:

    .. math::
        \\exp(i * \\pi/4 * \\sigma_{xx}) * \\exp(i * \\pi/8 * \\sigma_{yy})

    References:
        - https://arxiv.org/pdf/quant-ph/0312193.pdf
    """

    _num_qudits = 2
    _qasm_name = 'b'
    _expr = _UnitaryExpression(
        'B() { [[cos(pi/8),0,0,i*sin(pi/8)],'
        '[0,sin(pi/8),i*cos(pi/8),0],'
        '[0,i*cos(pi/8),sin(pi/8),0],'
        '[i*sin(pi/8),0,0,cos(pi/8)]] }',
    )
