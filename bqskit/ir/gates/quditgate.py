"""This module implements the QuditGate base class."""
from __future__ import annotations

from bqskit.ir.gate import Gate


class QuditGate(Gate):
    """A gate that acts on qudits."""

    _num_levels: int

    @property
    def radixes(self) -> tuple[int, ...]:
        """The number of orthogonal states for each qudit."""
        return tuple([self.num_levels] * self.num_qudits)

    @property
    def num_levels(self) -> int:
        """The number of levels in each qudit."""
        return getattr(self, '_num_levels')
