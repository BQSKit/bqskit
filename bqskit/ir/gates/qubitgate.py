"""This module implements the QubitGate base class."""
from __future__ import annotations

from bqskit.ir.gate import Gate


class QubitGate(Gate):
    """A gate that only acts on qubits."""

    @property
    def radixes(self) -> tuple[int, ...]:
        """The number of orthogonal states for each qudit."""
        return tuple([2] * self.num_qudits)
