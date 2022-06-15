"""
=============================================================
Circuit Intermediate Representation (:mod:`bqskit.ir`)
=============================================================

.. currentmodule:: bqskit.ir

The BQSKit Circuit structure is a 2-d array of operations. A circuit is
indexed first by cycles and second by qudits. The cycle index determines
when operations are executed, and the qudit index determines which
qudits an operation operates on.

Every component of the IR is also a function from a vector
real numbers to a unitary matrix. This is done to facilitate circuit
instantiatation, which is a very important primitive in quantum synthesis.

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
    CircuitStructure

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
from bqskit.ir.structure import CircuitStructure

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
    'CircuitStructure',
]
