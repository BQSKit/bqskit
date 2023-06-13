"""This module implements the ZZGate.""" 
from __future__ import annotations 

import math

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qutritgate import QutritGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.ir.gates.qutrit.constant.z import ZGate, Z0Gate, Z1Gate, Z2Gate
from bqskit.utils.math import kron


class ZZGate(ConstantGate, QutritGate):
    """
    The Ising ZZ coupling gate for qutrits
    """

    _num_qudits = 2
    _qasm_name = 'zz'
    
    def get_unitary(self) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(param)
        op=kron([ZGate._utry, ZGate._utry]).tolist()
        return UnitaryMatrix(op,)

    
class Z0Z0Gate(ConstantGate, QutritGate):
    """
    The Ising ZZ coupling gate for qutrits
    """

    _num_qudits = 2
    _qasm_name = 'zz'
    
    def get_unitary(self) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(param)
        op=kron([Z0Gate._utry, Z0Gate._utry]).tolist()
        return UnitaryMatrix(op,)

class Z1Z1Gate(ConstantGate, QutritGate):
    """
    The Ising ZZ coupling gate for qutrits
    """

    _num_qudits = 2
    _qasm_name = 'zz'
    
    def get_unitary(self) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(param)
        op=kron([Z1Gate._utry, Z1Gate._utry]).tolist()
        return UnitaryMatrix(op,)
    
class Z2Z2Gate(ConstantGate, QutritGate):
    """
    The Ising ZZ coupling gate for qutrits
    """

    _num_qudits = 2
    _qasm_name = 'zz'
    
    def get_unitary(self) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(param)
        op=kron([Z2Gate._utry, Z2Gate._utry]).tolist()
        return UnitaryMatrix(op,)