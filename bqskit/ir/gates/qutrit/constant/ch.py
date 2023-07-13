"""This module implements the CHGate.""" 
from __future__ import annotations

import math

from bqskit.ir.gates.constant.h import HGate
from bqskit.ir.gates.composed import ControlledGate

class CHGate(ControlledGate):
    """
    The controlled-Hadamard gate for an arbitrary qudit
    
    num_levels (int): the number of levels in the qudit

    controls list(int): The levels of control qudits.
    
    """

    _qasm_name = 'ch'
    _num_params = 0

    def __init__(self, controls: int, num_levels=2: int, level_1=0: int, level_2=1: int):    
        super(CHGate, self).__init__(HGate(num_levels=num_levels), num_levels=num_levels, controls=controls)