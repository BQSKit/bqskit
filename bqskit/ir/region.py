"""This module implements the CircuitRegion class."""
from __future__ import annotations

from typing import NamedTuple

from bqskit.ir.location import CircuitLocation


class CircuitRegion(NamedTuple):
    """
    The CircuitRegion class.

    A CircuitRegion is an ordered set of qudit indices. In context, this usually
    describes where a gate or operation is being applied.
    """
    qudits: CircuitLocation
    bounds: dict[int, tuple[int, int]]
