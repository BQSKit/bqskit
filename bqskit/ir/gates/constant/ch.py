"""This module implements the CHGate."""
from __future__ import annotations

from bqskit.ir.gates.composed import ControlledGate
from bqskit.ir.gates.constant.h import HGate
from typing import Sequence


class CHGate(ControlledGate):
    """
    The controlled-Hadamard gate for qudits.

    With default parameters, the CHGate is given by
    
    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & \\frac{\\sqrt{2}}{2} & \\frac{\\sqrt{2}}{2} \\\\
        0 & 0 & \\frac{\\sqrt{2}}{2} & -\\frac{\\sqrt{2}}{2} \\\\
        \\end{pmatrix}
    
    When parameters are changed, see ~ControlledGate for more info
    """

    _num_qudits = 2
    _qasm_name = 'ch'

    def __init__(
        self, 
        num_controls: int=1, 
        num_levels: Sequence[int] | int = 2, 
        level_of_each_control: Sequence[Sequence[int]] | None = None
    ) -> None:
        """Builds the CHGate, see :class:`ControlledGate` for more information."""
        super().__init__(
            HGate(num_levels=num_levels),
            num_controls=1,
            num_levels=num_levels,
            level_of_each_control = None
        )