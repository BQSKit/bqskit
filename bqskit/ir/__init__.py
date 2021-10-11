"""
=============================================================
Circuit Intermediate Representation (:mod:`bqskit.ir`)
=============================================================

.. currentmodule:: bqskit.ir

.. rubric:: Core Classes

.. autosummary::
    :toctree: autogen
    :recursive:

    Circuit
    Gate
    Operation

.. rubric:: Circuit Indexing and Helpers

.. autosummary::
    :toctree: autogen
    :recursive:

    CycleInterval
    CircuitIterator
    CircuitLocation
    CircuitPoint
    CircuitRegion

.. rubric:: Type Aliases

.. autosummary::
    :toctree: autogen
    :recursive:

    IntervalLike
    CircuitLocationLike
    CircuitPointLike
    CircuitRegionLike

.. automodule:: bqskit.ir.gates
   :no-members:
   :no-inherited-members:
   :no-special-members:

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
