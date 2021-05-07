"""
This module implements the QubitGate base class.

A QubitGate is one that only acts on qubits.

"""
from __future__ import annotations

from bqskit.ir.gate import Gate


class QubitGate(Gate):
    """The QubitGate Class."""

    def get_radixes(self) -> tuple[int, ...]:
        """Returns the number of orthogonal states for each qudit."""
        return tuple([2] * self.get_size())
