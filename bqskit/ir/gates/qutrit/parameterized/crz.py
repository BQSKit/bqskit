"""This module implements the CRZGate.""" #TODO convert to new format for qudit
from __future__ import annotations

import numpy as np
import numpy.typing as npt


from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.utils.cachedclass import CachedClass
from bqskit.ir.gates.parameterized import RZGate
from bqskit.ir.gates.composed import ControlledGate

class CRZGate(
    ControlledGate,
    DifferentiableUnitary,
    CachedClass,
):
    """
    A gate representing a controlled Z rotation for qudits
    
    __init__() arguments:
        num_levels : int
            Number of levels in each qudit (d).
        level: int
            The level on which to apply the RZ gate (0...d-1).
        controls: list(int)
            List of control levels

    get_unitary arguments:
                param: float
                The angle by which to rotate
    
    """

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'crz'
    
    
    def __init__(self, controls: int, num_levels=2: int, level_1=0: int, level_2=1: int):    
        super(CRZGate, self).__init__(RZGate(num_levels=num_levels, level=level), 
                                       num_levels=num_levels, controls=controls)