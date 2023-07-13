"""This module implements the CZGate.""" 
from __future__ import annotations

from bqskit.ir.gates.constant.z import ZGate
from bqskit.ir.gates.composed import ControlledGate
from bqskit.qis.unitary import IntegerVector


class CZGate(ControlledGate):
    """
    The Controlled-Z gate for qudits
    num_levels: (int) = number of levels os single qudit (greater or equal to 2)
    controls: list(int) = list of control levels
    level: (int) = level for (-1) phase
    """

    _qasm_name = 'cz'
    _num_params = 0

    def get_unitary(self, param: int ) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(param)
        op=cgate(param,ZGate._utry).tolist()
        return UnitaryMatrix(op,)
    
    def __init__(self, controls: IntegerVector =[1], num_levels: int=2, level: int=1):    
        super(CZGate, self).__init__(ZGate(num_levels=num_levels, level=level), 
                                       num_levels=num_levels, controls=controls)