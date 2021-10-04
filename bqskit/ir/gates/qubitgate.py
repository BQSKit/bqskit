"""
This module implements the QubitGate base class.

A QubitGate is one that only acts on qubits.
"""
from __future__ import annotations

from bqskit.ir.gate import Gate


class QubitGate(Gate):
    """The QubitGate Class."""

    @property
    def radixes(self) -> tuple[int, ...]:
        """The number of orthogonal states for each qudit."""
        return tuple([2] * self.num_qudits)
