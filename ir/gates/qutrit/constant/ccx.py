"""This module implements the CCXGate/ToffoliGate.""" #TODO adaprt for qutrit
from __future__ import annotations

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qutritgate import QutritGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.qis.unitary.unitary import IntegerVector
from bqskit.ir.gates.qutrit.constant.x import XGate, X01Gate, X02Gate, X12Gate
import itertools 
import numpy as np
from bqskit.ir.gates.qutrit import ccgate


class CCXGate(ConstantGate, QutritGate):
    """
    The toffoli gate, equal to an X gate with two controls
    User need to specify the state (0,1,2) of the control qutrit

    """

    _num_params = 2
    _num_qudits = 3
    _qasm_name = 'ccx'

    def get_unitary(self, params: IntegerVector = []) -> UnitaryMatrix:
        op=XGate._utry
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)
        op=ccgate(params,op).tolist()
        return UnitaryMatrix(op,)
    
class CCX01Gate(ConstantGate, QutritGate):
    """
    The CCX01 gate, equal to an X gate with two controls
    User need to specify the state (0,1,2) of the control qutrit
    """

    _num_params = 2
    _num_qudits = 3
    _qasm_name = 'ccx01'

    def get_unitary(self, params: IntegerVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)
        op=ccgate(params,X01Gate._utry).tolist()
        return UnitaryMatrix(op,)
    
class CCX02Gate(ConstantGate, QutritGate):
    """
    The CCX02 gate, equal to an X gate with two controls
    User need to specify the state (0,1,2) of the control qutrit
    """

    _num_params = 2
    _num_qudits = 3
    _qasm_name = 'ccx02'

    def get_unitary(self, params: IntegerVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)
        op=ccgate(params,X02Gate._utry).tolist()
        return UnitaryMatrix(op,)
    
class CCX12Gate(ConstantGate, QutritGate):
    """
    The CCX12 gate, equal to an X gate with two controls
    User need to specify the state (0,1,2) of the control qutrit
    """

    _num_params = 2
    _num_qudits = 3
    _qasm_name = 'ccx12'

    def get_unitary(self, params: IntegerVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)
        op=ccgate(params,X12Gate._utry).tolist()
        return UnitaryMatrix(op,)
        
ToffoliGate = CCXGate
