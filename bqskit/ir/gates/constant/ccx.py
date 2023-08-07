"""This module implements the CNOTGate/CXGate."""
from __future__ import annotations

from bqskit.ir.gates.composed import ControlledGate
from bqskit.ir.gates.constant.x import XGate
from typing import Sequence


class CCXGate(ControlledGate):
    """
    The CCX or Toffoli gate for qudits.
    """

    _qasm_name = 'ccx'
    _num_params = 0

    def __init__(
        self, 
        num_controls: int=2, 
        num_levels: Sequence[int] | int = 2, 
        level_of_each_control: Sequence[Sequence[int]] | None = None
    ) -> None:
        """Builds the CCXGate, see :class:`ControlledGate` for more information."""
        super().__init__(
            XGate(num_levels=num_levels, level_1=level_1, level_2=level_2),
            num_controls=2,
            num_levels=num_levels,
            level_of_each_control = None
        )

ToffoliGate = CCXGate
