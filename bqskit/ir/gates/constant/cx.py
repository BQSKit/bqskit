"""This module implements the CNOTGate/CXGate."""
from __future__ import annotations

from bqskit.ir.gates.composed import ControlledGate
from bqskit.ir.gates.constant.x import XGate
from typing import Sequence


class CNOTGate(ControlledGate):
    """
    The Controlled-Not or Controlled-X gate for qudits.
    """

    _qasm_name = 'cx'
    _num_params = 0

    def __init__(
        self, 
        num_controls: int=1, 
        num_levels: Sequence[int] | int = 2, 
        level_1: int=0,
        level_2: int=1,
        level_of_each_control: Sequence[Sequence[int]] | None = None
    ) -> None:
        """Builds the CNOTGate, see :class:`ControlledGate` for more information."""
        super().__init__(
            XGate(num_levels=num_levels, level_1=level_1, level_2=level_2),
            num_controls=num_controls,
            num_levels=num_levels,
            level_of_each_control = level_of_each_control
        )


CXGate = CNOTGate