"""
This module implements the CircuitCell class.

A Circuit is composed of a grid of CircuitCells.
A cell groups together a gate and its parameters.
"""
from __future__ import annotations

from typing import Iterable
from typing import Sequence

from bqskit.ir.gate import Gate


class CircuitCell():
    """The CircuitCell class."""

    def __init__(
        self, gate: Gate,
        location: Iterable[int],
        params: Sequence[float] | None = None,
    ) -> None:
        """
        CircuitCell Constructor.
s
        Args:
            gate (Gate): The cell's gate.

            location (Iterable[int]):  The set of qudits this gate affects.

            params (Optional[Sequence[float]]): The parameters for the
                gate, if any.
        """
        self.gate = gate
        self.location = location
        self.params = params
    
    def get_qasm(self) -> str:
        """Returns the qasm string for this operation."""
        return "%s(%s) q[%s];" % ( self.gate.get_qasm_name(),
                                   ", ".join(self.params),
                                   "], q[".join(self.location))

    def __str__(self) -> str:
        pass  # TODO
    
    def __repr__(self) -> str:
        pass  # TODO