"""
This module implements the CircuitCell class.

A Circuit is composed of a grid of CircuitCells.
A cell groups together a gate and its parameters.
"""
from typing import List
from typing import Optional

from bqskit.ir.gate import Gate


class CircuitCell():
    """The CircuitCell class."""

    def __init__(
        self, gate: Gate, params: Optional[List[float]] = None,
        qudit_idx: int = 0,
    ):
        """
        CircuitCell Constructor.
s
        Args:
            gate (Gate): The gate in this Cell.

            params (List[float]): The parameters for the gate, if any.

            qudit_idx (int): This cell's qudit index in the gate.
        """
        self.gate = gate
        self.params = params
        self.qudit_idx = qudit_idx
