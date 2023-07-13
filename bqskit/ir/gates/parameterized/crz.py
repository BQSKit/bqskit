"""This module implements the CRZGate."""
from __future__ import annotations

from bqskit.ir.gates.quditgate import QuditGate
from bqskit.ir.gates.parameterized.rz import RZGate
from bqskit.qis.unitary import IntegerVector
from bqskit.ir.gates.composed import ControlledGate

class CRZGate(ControlledGate):
    """
    A gate representing a controlled Z rotation for qudits.

    This is equivalent to a controlled rotation by the X Pauli Gate in the subspace of 2 levels
    
        __init__() arguments:
            num_levels : int
                Number of levels in each qudit (d).
            level_1,level_2: int
                 The levels on which to apply the Z gate (0...d-1).
                        
        get_unitary arguments:
                param: float
                The angle by which to rotate

    """

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'crz'
    
    def __init__(self, num_levels: int=2, controls: IntegerVector=[1], level_1: int=0, level_2: int=1):    
        super(CRZGate, self).__init__(RZGate(num_levels=num_levels, level_1=level_1, level_2=level_2), 
                                       num_levels=num_levels, controls=controls)