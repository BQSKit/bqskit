"""This module implements the CircuitRegion class."""
from __future__ import annotations

from typing import NamedTuple

from bqskit.ir.location import CircuitLocation
from bqskit.ir.point import CircuitPoint


class CircuitRegion(NamedTuple):
    """
    The CircuitRegion class.

    A CircuitRegion is an contiguous area in a circuit.
    """
    qudits: CircuitLocation
    bounds: dict[int, tuple[int, int]]

    def overlaps(self, other: CircuitPoint | CircuitRegion) -> bool:
        """Return true if `other` overlaps this region."""
        pass  # TODO
