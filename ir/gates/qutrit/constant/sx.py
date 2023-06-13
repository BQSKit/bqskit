"""This module implements the SqrtXGate/SXGate."""
from __future__ import annotations

import scipy

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qutrit.constant.x import XGate
from bqskit.ir.gates.qutritgate import QutritGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class SqrtXGate(ConstantGate, QutritGate):
    """The Sqrt(X) gate for qutrit."""

    _num_qudits = 1
    _qasm_name = 'sx'
    op = XGate._utry
    _utry = UnitaryMatrix(scipy.linalg.sqrtm(op).tolist())


SXGate = SqrtXGate
