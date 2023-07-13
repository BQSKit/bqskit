"""This module implements the XXGate."""
from __future__ import annotations

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qutrit.constant.x import X01Gate
from bqskit.ir.gates.qutrit.constant.x import X02Gate
from bqskit.ir.gates.qutrit.constant.x import X12Gate
from bqskit.ir.gates.qutrit.constant.x import XGate
from bqskit.ir.gates.qutritgate import QutritGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.math import kron


class XXGate(ConstantGate, QutritGate, UnitaryMatrix):
    """The Ising XX coupling gate for qutrits."""

    _num_qudits = 2
    _qasm_name = 'xx'
    _utry = UnitaryMatrix(kron([XGate._utry, XGate._utry]).tolist())


class X01X01Gate(ConstantGate, QutritGate, UnitaryMatrix):
    """The X01X01 coupling gate for qutrits."""

    _num_qudits = 2
    _qasm_name = 'x01x01'
    _utry = UnitaryMatrix(kron([X01Gate._utry, X01Gate._utry]).tolist())


class X02X02Gate(ConstantGate, QutritGate, UnitaryMatrix):
    """The X02X02 coupling gate for qutrits."""

    _num_qudits = 2
    _qasm_name = 'x02x02'
    _utry = UnitaryMatrix(kron([X02Gate._utry, X02Gate._utry]).tolist())


class X01X02Gate(ConstantGate, QutritGate, UnitaryMatrix):
    """The X01X02 coupling gate for qutrits."""

    _num_qudits = 2
    _qasm_name = 'x01x02'
    _utry = UnitaryMatrix(kron([X01Gate._utry, X02Gate._utry]).tolist())


class X02X01Gate(ConstantGate, QutritGate, UnitaryMatrix):
    """The X02X01 coupling gate for qutrits."""

    _num_qudits = 2
    _qasm_name = 'x02x01'
    _utry = UnitaryMatrix(kron([X02Gate._utry, X01Gate._utry]).tolist())
