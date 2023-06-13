"""This module implements the CSGate.""" #TODO adapt for qutrit
from __future__ import annotations

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qutritgate import QutritGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.ir.gates.qutrit.constant.s import SGate
from bqskit.ir.gates.qutrit import cgate


class CSGate(ConstantGate, QutritGate):
    """
    The Controlled-S gate for qutrit
    User need to specify the state (0,1,2) of the control qubit

    """

    _num_qudits = 2
    _qasm_name = 'cs'
    _num_params = 1

    def get_unitary(self, param: int ) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(param)
        op=cgate(param,SGate._utry).tolist()
        return UnitaryMatrix(op,)
    