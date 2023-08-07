"""This module implements the CYGate."""
from __future__ import annotations

from bqskit.ir.gates.composed import ControlledGate
from bqskit.ir.gates.constant.y import YGate
from typing import Sequence


class CYGate(ControlledGate): 
    """
    The Controlled-Y gate for qudits.
    """

    _qasm_name = 'cy'
    _num_params = 0

    def __init__(
        self, 
        num_controls: int=1, 
        num_levels: Sequence[int] | int = 2, 
        level_of_each_control: Sequence[Sequence[int]] | None = None
    ) -> None:
        """Builds the CYGate, see :class:`ControlledGate` for more information."""
        super().__init__(
            YGate(num_levels=num_levels, level_1=level_1, level_2=level_2),
            num_controls=num_controls,
            num_levels=num_levels,
            level_of_each_control = level_of_each_control
        )