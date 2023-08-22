"""This module implements the QuditGate base class."""
from __future__ import annotations

from bqskit.ir.gate import Gate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class QuditGate(Gate):
    """A gate that acts on qudits."""

    _num_levels: int
    _radixes: tuple[int, ...] 

    @property
    def radixes(self) -> tuple[int, ...]:
        """The number of orthogonal states for each qudit."""
        if hasattr(self, '_radixes'):
            return getattr(self, '_radixes')
        else:
            self._radixes = tuple([self.num_levels] * self.num_qudits)
            return getattr(self, '_radixes') 

    @property
    def num_levels(self) -> int:
        """The number of levels in each qudit."""
        return getattr(self, '_num_levels')
