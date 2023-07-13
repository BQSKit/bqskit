"""This module implements the CPGate."""
from __future__ import annotations

from bqskit.ir.gates.composed import ControlledGate
from bqskit.ir.gates.parameterized.p import PGate
from bqskit.qis.unitary import IntegerVector


class CPGate(ControlledGate):
    """
    A gate representing a controlled phase rotation for qudits.

    __init__() arguments:
        num_levels : int
            Number of levels in each qudit (d).
        level : int
             The levels on which to apply the Phase gate (0...d-1).

    get_unitary arguments:
            param: float
            The angle by which to rotate
    """

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'cp'

    def __init__(self, num_levels: int = 2, controls: IntegerVector = [1], level: int = 1):
        super().__init__(
            PGate(num_levels=num_levels, level=level),
            num_levels=num_levels, controls=controls,
        )
