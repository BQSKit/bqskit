"""This module implements the TGate."""
from __future__ import annotations

import cmath

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
import numpy as np

class TGate(ConstantGate, QubitGate):
    """
    The single-qubit T gate (Z^(1/4)) for qutrits

    """

    _num_qudits = 1
    _qasm_name = 't'
    _w=np.exp(2*np.pi*1j/3)
    _utry = UnitaryMatrix(
   [[1,0,0],
    [0,_w**(0.25),0],
    [0,0,_w**(0.5)]],[3])
