"""This module implements the SqrtCNOTGate."""
from __future__ import annotations

from bqskit.ir.gates.composed import ControlledGate
from bqskit.ir.gates.constant.sx import SqrtXGate
from typing import Sequence


class SqrtCNOTGate(ControlledGate):
    """
    The Square root Controlled-X gate for qudits.

    """

    _num_qudits = 2

    def __init__(
        self, 
        num_controls: int=1, 
        num_levels: Sequence[int] | int = 2, 
        level_of_each_control: Sequence[Sequence[int]] | None = None
    ) -> None:
        """Builds the SqrtCNOTGate, see :class:`ControlledGate` for more information."""
        super().__init__(
            SqrtXGate(num_levels=num_levels),
            num_controls=1,
            num_levels=num_levels,
            level_of_each_control = None
        )
