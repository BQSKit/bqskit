"""This module implements the CRXGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.utils.cachedclass import CachedClass
from bqskit.ir.gates.parameterized import RXGate
from bqskit.ir.gates.composed import ControlledGate

class CRXGate(
    ControlledGate,
    DifferentiableUnitary,
    CachedClass,
):
    """
    A gate representing a controlled X rotation for qudits
    
    __init__() arguments:
        num_levels : int
            Number of levels in each qudit (d).
        level_1,level_2: int
            The levels on which to apply the RX gate (0...d-1).
        controls: list(int)
            List of control levels

    get_unitary arguments:
                param: float
                The angle by which to rotate
    
    """

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'crx'
    
    
    def __init__(self, controls: int, num_levels=2: int, level_1=0: int, level_2=1: int):    
        super(CNOTGate, self).__init__(RXGate(num_levels=num_levels, level_1=level_1, level_2=level_2), 
                                       num_levels=num_levels, controls=controls)