"""This module implements the CNOTGate/CXGate.""" #TODO check kronecker product order
from __future__ import annotations

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qutritgate import QutritGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.ir.gates.qutrit.constant.x import XGate, X01Gate, X02Gate, X12Gate
from bqskit.ir.gates.qutrit import cgate


class CNOTGate(ConstantGate, QutritGate):
    """
    The Controlled-Not or Controlled-X gate for qutrits
    User need to specify the state (0,1,2) of the control qutrit
    """

    _num_qudits = 2
    _qasm_name = 'cx'
    _num_params = 1

    def get_unitary(self, param: int ) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(param)
        op=cgate(param,XGate._utry).tolist()
        return UnitaryMatrix(op,)

CXGate = CNOTGate

class CX01Gate(ConstantGate, QutritGate):
    """
    The Controlled-X01 gate for qutrits
    User need to specify the state (0,1,2) of the control qutrit
    """

    _num_qudits = 2
    _qasm_name = 'cx01'
    _num_params = 1
    
    def get_unitary(self, param: int ) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(param)
        op=cgate(param,X01Gate._utry).tolist()
        return UnitaryMatrix(op,) 
       
    
class CX02Gate(ConstantGate, QutritGate):
    """
    The Controlled-X02 gate for qutrits
    User need to specify the state (0,1,2) of the control qutrit
    """

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'cx02'
    
    def get_unitary(self, param: int ) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(param)
        op=cgate(param,X02Gate._utry).tolist()
        return UnitaryMatrix(op,)
    
class CX12Gate(ConstantGate, QutritGate):
    """
    The Controlled-X12 gate for qutrits
    User need to specify the state (0,1,2) of the control qutrit
    """

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'cx12'
    
    def get_unitary(self, param: int ) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(param)
        op=cgate(param,X02Gate._utry).tolist()
        return UnitaryMatrix(op,)
    


