"""This module implements the BGate."""
from __future__ import annotations

from numpy import pi
from openqudit.expressions import UnitaryExpression as _UnitaryExpression
from scipy.linalg import expm

from bqskit.ir.gate import Gate
from bqskit.qis.pauli import PauliMatrices
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
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

    def __init__(self) -> None:
        """Construct a BGate."""
        paulis = PauliMatrices(2)
        xx = paulis[5]
        yy = paulis[10]
        mat = expm(1j * pi / 4 * xx) @ expm(1j * pi / 8 * yy)
        self._utry = UnitaryMatrix(mat)
