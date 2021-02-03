"""
This module implements the CircuitCell class.

A Circuit is composed of a grid of CircuitCells.
A cell groups together a gate and its parameters.
"""
from __future__ import annotations

from typing import Iterable
from typing import Sequence


class CircuitCell():
    """The CircuitCell class."""

    def __init__(
        self, gate_index: int,
        location: Iterable[int],
        params: Sequence[float] | None = None,
    ) -> None:
        """
        CircuitCell Constructor.
s
        Args:
            gate_index (int): The index in the circuit's gate set
                that determines the gate in this cell.

            location:  The set of qudits this gate affects.

            params (Optional[Sequence[float]]): The parameters for the
                gate, if any.
        """
        self.gate_index = gate_index
        self.location = location
        self.params = params
