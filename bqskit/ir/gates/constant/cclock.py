"""This module implements the CPDGate."""
from __future__ import annotations

from bqskit.ir.gates.composed import ControlledGate
from bqskit.ir.gates.constant.clock import ClockGate


class CClockGate(ControlledGate):
    """
    The Controlled-Clock gate for qudits.
    """

    #_num_qudits = 2
    def __init__(self, num_controls: int=1, num_levels: Sequence[int] = [2] | int = 2, level_of_each_control: Sequence[Sequence[int]] = [[1]]):
        super().__init__(
            ClockGate(num_levels=num_levels),
            num_levels=num_levels, controls=controls,
        )
