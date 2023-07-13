"""This module implements the CHGate."""  # TODO adapt for qutrit
from __future__ import annotations

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qutrit import cgate
from bqskit.ir.gates.qutrit.constant.h import HGate
from bqskit.ir.gates.qutritgate import QutritGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class CHGate(ConstantGate, QutritGate):
    """
    The controlled-Hadamard gate for qutrit.

    User need to specify the state (0,1,2) of the control qutrit
    """

    _num_qudits = 2
    _qasm_name = 'ch'
    _num_params = 1

    def get_unitary(self, param: int) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(param)
        op = cgate(param, HGate._utry).tolist()
        return UnitaryMatrix(op)
