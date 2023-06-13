"""This module implements the CZGate."""  # TODO adapt for qutrit
from __future__ import annotations

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qutrit import cgate
from bqskit.ir.gates.qutrit.constant.z import Z0Gate
from bqskit.ir.gates.qutrit.constant.z import Z1Gate
from bqskit.ir.gates.qutrit.constant.z import Z2Gate
from bqskit.ir.gates.qutrit.constant.z import ZGate
from bqskit.ir.gates.qutritgate import QutritGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class CZGate(ConstantGate, QutritGate):
    """
    The Controlled-Z gate.

    User need to specify the state (0,1,2) of the control qutrit
    """

    _num_qudits = 2
    _qasm_name = 'cz'
    _num_params = 1

    def get_unitary(self, param: int) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(param)
        op = cgate(param, ZGate._utry).tolist()
        return UnitaryMatrix(op)


class CZ0Gate(ConstantGate, QutritGate):
    """
    The Controlled-Z0 gate.

    User need to specify the state (0,1,2) of the control qutrit
    """

    _num_qudits = 2
    _qasm_name = 'cz0'
    _num_params = 1

    def get_unitary(self, param: int) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(param)
        op = cgate(param, Z0Gate._utry).tolist()
        return UnitaryMatrix(op)


class CZ1Gate(ConstantGate, QutritGate):
    """
    The Controlled-Z1 gate.

    User need to specify the state (0,1,2) of the control qutrit
    """

    _num_qudits = 2
    _qasm_name = 'cz0'
    _num_params = 1

    def get_unitary(self, param: int) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(param)
        op = cgate(param, Z1Gate._utry).tolist()
        return UnitaryMatrix(op)


class CZ2Gate(ConstantGate, QutritGate):
    """
    The Controlled-Z0 gate.

    User need to specify the state (0,1,2) of the control qutrit
    """

    _num_qudits = 2
    _qasm_name = 'cz0'
    _num_params = 1

    def get_unitary(self, param: int) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(param)
        op = cgate(param, Z2Gate._utry).tolist()
        return UnitaryMatrix(op)
