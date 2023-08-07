"""This module implements the CZGate."""
from __future__ import annotations

from bqskit.ir.gates.composed import ControlledGate
from bqskit.ir.gates.constant.z import ZGate
from typing import Sequence


class CZGate(ControlledGate):
    """
    The Controlled-Z gate for qudits. 
    """

    _qasm_name = 'cz'
    _num_params = 0

    def __init__(
        self, 
        num_controls: int=1, 
        num_levels: Sequence[int] | int = 2, 
        level_of_each_control: Sequence[Sequence[int]] | None = None
    ) -> None:
        """Builds the CZGate, see :class:`ControlledGate` for more information."""
        super().__init__(
            ZGate(num_levels=num_levels, level_1=level_1, level_2=level_2),
            num_controls=1,
            num_levels=num_levels,
            level_of_each_control = None
        )