"""This module implements the QutritGate base class."""
from __future__ import annotations

from bqskit.ir.gate import Gate


class QutritGate(Gate):
    """A gate that only acts on qutrits."""

    @property
    def radixes(self) -> tuple[int, ...]:
        """The number of orthogonal states for each qudit."""
        return tuple([3] * self.num_qudits)
