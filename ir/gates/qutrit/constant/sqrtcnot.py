"""This module implements the SqrtCNOTGate."""
from __future__ import annotations

import scipy

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qutrit.constant.cx import CXGate
from bqskit.ir.gates.qutritgate import QutritGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class SqrtCXGate(ConstantGate, QutritGate):
    """The Square root Controlled-X gate for qutrits."""

    _num_qudits = 2
    _qasm_name = 'csx'
    _utry = UnitaryMatrix(scipy.linalg.sqrtm(CXGate._utry))
