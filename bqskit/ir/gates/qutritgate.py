"""
This module implements the QutritGate base class.

A QutritGate is one that only acts on qutrits.
"""
from __future__ import annotations

from bqskit.ir.gate import Gate


class QutritGate(Gate):
    """The QutritGate Class."""

    @property
    def radixes(self) -> tuple[int, ...]:
        """The number of orthogonal states for each qudit."""
        return tuple([3] * self.num_qudits)
