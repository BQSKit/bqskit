"""
This module implements the QubitGate base class.

A QubitGate is one that only acts on qubits.
"""

from __future__ import annotations

from bqskit.ir.gate import Gate

class QubitGate(Gate):
    """The QubitGate Class."""

    def get_radixes(self) -> list[int]:
        """Returns the number of orthogonal states for each qudit."""
        return [2] * self.get_size()
