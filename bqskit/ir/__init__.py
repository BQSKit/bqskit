"""
=============================================================
BQSKit Circuit Intermediate Representation (:mod:`bqskit.ir`)
=============================================================

TODO: More description
"""
from __future__ import annotations

from bqskit.ir.circuit import Circuit
from bqskit.ir.gate import Gate
from bqskit.ir.interval import CycleInterval
from bqskit.ir.interval import IntervalLike
from bqskit.ir.iterator import CircuitIterator
from bqskit.ir.location import CircuitLocation
from bqskit.ir.location import CircuitLocationLike
from bqskit.ir.operation import Operation
from bqskit.ir.point import CircuitPoint
from bqskit.ir.point import CircuitPointLike
from bqskit.ir.region import CircuitRegion
from bqskit.ir.region import CircuitRegionLike

__all__ = [
    'Operation',
    'Circuit',
    'Gate',
    'CircuitIterator',
    'CircuitLocation',
    'CircuitLocationLike',
    'CircuitPoint',
    'CircuitPointLike',
    'CircuitRegion',
    'CircuitRegionLike',
    'CycleInterval',
    'IntervalLike',
]
