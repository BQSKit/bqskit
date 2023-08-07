"""This module implements the CPGate."""
from __future__ import annotations

from bqskit.ir.gates.composed import ControlledGate
from bqskit.ir.gates.parameterized.p import PGate
from typing import Sequence


class CPGate(ControlledGate):
    """
    A gate representing a controlled phase rotation for qudits.

    For qubits the gates is represented bythe following matrix:
    
    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & 1 & 0 \\\\
        0 & 0 & 0 & \\exp({i\\theta}) \\\\
        \\end{pmatrix}

     """

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'cp'

    def __init__(
        self, 
        num_controls: int=1, 
        num_levels: Sequence[int] | int = 2, 
        level: int=1,
        level_of_each_control: Sequence[Sequence[int]] | None = None
    ) -> None:
        """Builds the CHGate, see :class:`ControlledGate` for more information."""
        super().__init__(
            PGate(num_levels=num_levels,level=level),
            num_controls=1,
            num_levels=num_levels,
            level_of_each_control = None
        )