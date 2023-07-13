"""This module implements the CPDGate."""
from __future__ import annotations

from bqskit.ir.gates.constant.clock import ClockGate
from bqskit.ir.gates.composed import ControlledGate


class CClockGate(ControlledGate):
    """
    The Controlled-Clock gate for qudits

    num_levels: (int) = number of levels os single qudit (greater or equal to 2)
    controls: list(int) = list of control levels

    """

    _num_qudits = 2
    
    def __init__(self, num_levels: int, controls: IntegerVector):    
        super(CClockGate, self).__init__(ClockGate(num_levels=num_levels), 
                                       num_levels=num_levels, controls=controls)