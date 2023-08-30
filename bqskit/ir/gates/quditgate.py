"""This module implements the QuditGate base class."""
from __future__ import annotations

from typing import Sequence
from bqskit.ir.gate import Gate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_sequence_of_int


class QuditGate(Gate):
    """A gate that acts on qudits."""

    _num_levels: int | Sequence[int]

    @property
    def radixes(self) -> tuple[int, ...]:
        """The number of orthogonal states for each qudit."""
        if hasattr(self, '_radixes'):
            return getattr(self, '_radixes')
        else:
            if is_integer(self._num_levels):
                self._radixes = tuple([self.num_levels] * self.num_qudits)
            elif is_sequence_of_int(self._num_levels):
                self._radixes = tuple(self._num_levels)
            return getattr(self, '_radixes')

    @property
    def num_levels(self) -> int:
        """The number of levels in each qudit."""
        return getattr(self, '_num_levels')
