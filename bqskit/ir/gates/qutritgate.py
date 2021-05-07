"""
This module implements the QutritGate base class.

A QutritGate is one that only acts on qutrits.

"""
from __future__ import annotations

from bqskit.ir.gate import Gate


class QutritGate(Gate):
    """The QutritGate Class."""

    def get_radixes(self) -> tuple[int, ...]:
        """Returns the number of orthogonal states for each qudit."""
        return tuple([3] * self.get_size())
