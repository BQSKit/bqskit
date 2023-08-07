"""This module implements the CRXGate."""
from __future__ import annotations

from bqskit.ir.gates.composed import ControlledGate
from bqskit.ir.gates.parameterized.rx import RXGate
from typing import Sequence


class CRXGate(ControlledGate):
    """
    A gate representing a controlled X rotation for qudits.

    This is equivalent to a controlled rotation by the X Pauli Gate in the subspace of 2 levels

        __init__() arguments:
            num_levels : int
                Number of levels in each qudit (d).
            level_1,level_2: int
                 The levels on which to apply the X gate (0...d-1).

        get_unitary arguments:
                param: float
                The angle by which to rotate
    """

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'crx'

    def __init__(
        self, 
        num_controls: int=1, 
        num_levels: Sequence[int] | int = 2, 
        level_1: int=0,
        level_2: int=1,
        level_of_each_control: Sequence[Sequence[int]] | None = None
    ) -> None:
        """Builds the CRXGate, see :class:`ControlledGate` for more information."""     
        super().__init__(
            RXGate(num_levels=num_levels, level_1=level_1, level_2=level_2),
            num_controls=num_controls,
            num_levels=num_levels,
            level_of_each_control = level_of_each_control
        )
