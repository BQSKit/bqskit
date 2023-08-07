"""This module implements the CHGate."""
from __future__ import annotations

from bqskit.ir.gates.composed import ControlledGate
from bqskit.ir.gates.constant.s import SGate
from typing import Sequence


class CSGate(ControlledGate):
    """
    The controlled-S gate for qudits
    """

    _num_qudits = 2
    _qasm_name = 'cs'

    def __init__(
        self, 
        num_controls: int=1, 
        num_levels: Sequence[int] | int = 2, 
        level_1: int=0,
        level_2: int=1,
        level_of_each_control: Sequence[Sequence[int]] | None = None
    ) -> None:
        """Builds the CSGate, see :class:`ControlledGate` for more information."""
        super().__init__(
            SGate(num_levels=num_levels, level_1=level_1, level_2=level_2),
            num_controls=num_controls,
            num_levels=num_levels,
            level_of_each_control = level_of_each_control
        )
