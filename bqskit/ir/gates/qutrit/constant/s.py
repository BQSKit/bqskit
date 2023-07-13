"""This module implements the SGate."""
from __future__ import annotations

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qutritgate import QutritGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
import numpy as np

class SGate(ConstantGate, QutritGate):
    """
    The single-qubit S gate.

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 \\\\
        0 & 1 & 0 \\\\
        0 & 0 & w\\end{pmatrix}
    """

    _num_qudits = 1
    _qasm_name = 's'
    _w=np.exp(2*np.pi*1j/3)
    _utry = UnitaryMatrix(
   [[1,0,0],
    [0,1,0],
    [0,0,_w]],[3])
