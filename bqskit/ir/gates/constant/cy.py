"""This module implements the CYGate."""
from __future__ import annotations

from bqskit.ir.gates.composed import ControlledGate
from bqskit.ir.gates.constant.y import YGate
from typing import Sequence


class CYGate(ControlledGate):
    """
    The Controlled-Not or Controlled-Y gate for qudits.

    num_levels: (int) = number of levels os single qudit (greater or equal to 2)
    controls: list(int) = list of control levels
    level_1: (int) = first level for Y gate
    level_2: (int) = second level for Y gate
    """

    _qasm_name = 'cy'
    _num_params = 0

    def __init__(self, num_levels: int = 2, controls: Sequence[int] = [1], level_1: int = 0, level_2: int = 1):
        super().__init__(
            YGate(num_levels=num_levels, level_1=level_1, level_2=level_2),
            num_levels=num_levels, controls=controls,
        )
