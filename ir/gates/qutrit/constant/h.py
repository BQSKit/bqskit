"""This module implements the HGate."""
from __future__ import annotations

import math

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qutritgate import QutritGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
import numpy as np

class HGate(ConstantGate, QutritGate):
    """
    The Hadamard gate.

    The H gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 1 & 1 \\\\
        1 & w & w^2 \\\\
        1 & w^2 & w\\end{pmatrix}
    """

    _num_qudits = 1
    _qasm_name = 'h'
    _w=np.exp(2*np.pi*1j/3)
    _utry = UnitaryMatrix(
   1/np.sqrt(3)*np.array([[1,1,1],
    [1,_w,_w**2],
    [1,_w**2,_w]]),[3])