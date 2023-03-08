"""This module implements the BGate."""
from __future__ import annotations

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.qis.pauli import PauliMatrices
from scipy.linalg import expm
from numpy import pi

class BGate(ConstantGate, QubitGate):
    """
    The 2 qubit B gate.

    The B gate is given by the following unitary:

    .. math::
        \\exp(i * \\pi/4 * \\sigma_{xx}) * \\exp(i * \\pi/8 * \\sigma_{yy})
    
    Unitary expression taken from: https://arxiv.org/pdf/quant-ph/0312193.pdf
    """
    _num_qudits = 2
    _qasm_name  = 'b'

    def __init__(self):
        paulis = PauliMatrices(2)
        xx = paulis[5]
        yy = paulis[10]
        self._utry = UnitaryMatrix(expm(1j * pi/4 * xx) @ expm(1j * pi/8 * yy))

