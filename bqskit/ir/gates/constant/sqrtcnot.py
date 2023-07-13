"""This module implements the SqrtCNOTGate."""
from __future__ import annotations
   
from bqskit.ir.gates.constant.sx import SqrtXGate
from bqskit.ir.gates.composed import ControlledGate


class SqrtCNOTGate(ControlledGate):
    """
    The Square root Controlled-X gate for qudits

    num_levels: (int) = number of levels os single qudit (greater or equal to 2)
    controls: list(int) = list of control levels

    """

    _num_qudits = 2
    
    def __init__(self, num_levels: int, controls: IntegerVector):    
        super(SqrtCNOTGate, self).__init__(SqrtXGate(num_levels=num_levels), 
                                       num_levels=num_levels, controls=controls)