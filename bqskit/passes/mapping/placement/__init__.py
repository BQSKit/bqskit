"""
This subpackage implements Placement algorithms.

Placement algorithms are responsible for selecting which set of physical qubits
to start the circuit on.
"""
from __future__ import annotations

from bqskit.passes.mapping.placement.greedy import GreedyPlacementPass
from bqskit.passes.mapping.placement.trivial import TrivialPlacementPass

__all__ = ['GreedyPlacementPass', 'TrivialPlacementPass']
