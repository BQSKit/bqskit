"""This module implements the CTGate.""" 
from __future__ import annotations

import cmath

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qutritgate import QutritGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.ir.gates.qutrit.constant.t import TGate
from bqskit.ir.gates.qutrit import cgate

class CTGate(ConstantGate, QutritGate):
    """
    The Controlled-T gate

    """

    _num_qudits = 2
    _qasm_name = 'ct'
    _num_params = 1

    def get_unitary(self, param: int ) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(param)
        op=cgate(param,TGate._utry).tolist()
        return UnitaryMatrix(op,)

