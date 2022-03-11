"""
====================================================================
Compilation and Synthesis Algorithms (:mod:`bqskit.passes`)
====================================================================

.. rubric:: Partitioning Passes

These passes will group together gates in the circuit. It is recommended
to use the QuickPartitioner currently.

.. autosummary::
    :toctree: autogen
    :recursive:

    ClusteringPartitioner
    GreedyPartitioner
    ScanPartitioner
    QuickPartitioner

.. rubric:: Synthesis Passes

These passes perform synthesis. Decomposition passes break down
unitaries into smaller ones and will need to be followed up by another
synthesis pass to convert the circuit to native gates.

.. autosummary::
    :toctree: autogen
    :recursive:

    LEAPSynthesisPass
    QSearchSynthesisPass
    QFASTDecompositionPass
    QPredictDecompositionPass

.. rubric:: Processing Passes

.. autosummary::
    :toctree: autogen
    :recursive:

    ScanningGateRemovalPass
    WindowOptimizationPass
    SubstitutePass

.. rubric:: Control Passes

This are passes that are used to manage control flow inside of a
compilation task.

.. autosummary::
    :toctree: autogen
    :recursive:

    DoWhileLoopPass
    ForEachBlockPass
    IfThenElsePass
    WhileLoopPass
    DoThenDecide
    ParallelDo

.. rubric:: Predicates

This objects are designed as conditions for use with control passes.

.. autosummary::
    :toctree: autogen
    :recursive:

    PassPredicate
    ChangePredicate
    GateCountPredicate
    NotPredicate

.. rubric:: Layout Passes

.. autosummary::
    :toctree: autogen
    :recursive:

    SimpleLayoutPass

.. rubric:: Utility Passes

.. autosummary::
    :toctree: autogen
    :recursive:

    CompressPass
    RecordStatsPass
    SetRandomSeedPass
    UnfoldPass
    UpdateDataPass
    ToU3Pass
    BlockConversionPass

.. rubric:: Search Heuristics

.. autosummary::
    :toctree: autogen
    :recursive:

    HeuristicFunction
    AStarHeuristic
    GreedyHeuristic
    DijkstraHeuristic

.. rubric:: Search Layer Generators

.. autosummary::
    :toctree: autogen
    :recursive:

    LayerGenerator
    SimpleLayerGenerator
    SeedLayerGenerator
    StairLayerGenerator
"""
from __future__ import annotations

from bqskit.passes.control.dothendecide import DoThenDecide
from bqskit.passes.control.dowhileloop import DoWhileLoopPass
from bqskit.passes.control.foreach import ForEachBlockPass
from bqskit.passes.control.ifthenelse import IfThenElsePass
from bqskit.passes.control.paralleldo import ParallelDo
from bqskit.passes.control.predicate import PassPredicate
from bqskit.passes.control.predicates.change import ChangePredicate
from bqskit.passes.control.predicates.count import GateCountPredicate
from bqskit.passes.control.predicates.notpredicate import NotPredicate
from bqskit.passes.control.whileloop import WhileLoopPass
from bqskit.passes.layout.simple import SimpleLayoutPass
from bqskit.passes.partitioning.cluster import ClusteringPartitioner
from bqskit.passes.partitioning.greedy import GreedyPartitioner
from bqskit.passes.partitioning.quick import QuickPartitioner
from bqskit.passes.partitioning.scan import ScanPartitioner
from bqskit.passes.processing.scan import ScanningGateRemovalPass
from bqskit.passes.processing.substitute import SubstitutePass
from bqskit.passes.processing.window import WindowOptimizationPass
from bqskit.passes.search.frontier import Frontier
from bqskit.passes.search.generator import LayerGenerator
from bqskit.passes.search.generators.seed import SeedLayerGenerator
from bqskit.passes.search.generators.simple import SimpleLayerGenerator
from bqskit.passes.search.generators.stair import StairLayerGenerator
from bqskit.passes.search.heuristic import HeuristicFunction
from bqskit.passes.search.heuristics.astar import AStarHeuristic
from bqskit.passes.search.heuristics.dijkstra import DijkstraHeuristic
from bqskit.passes.search.heuristics.greedy import GreedyHeuristic
from bqskit.passes.synthesis.leap import LEAPSynthesisPass
from bqskit.passes.synthesis.qfast import QFASTDecompositionPass
from bqskit.passes.synthesis.qpredict import QPredictDecompositionPass
from bqskit.passes.synthesis.qsearch import QSearchSynthesisPass
from bqskit.passes.synthesis.synthesis import SynthesisPass
from bqskit.passes.util.compress import CompressPass
from bqskit.passes.util.conversion import BlockConversionPass
from bqskit.passes.util.converttou3 import ToU3Pass
from bqskit.passes.util.random import SetRandomSeedPass
from bqskit.passes.util.record import RecordStatsPass
from bqskit.passes.util.unfold import UnfoldPass
from bqskit.passes.util.update import UpdateDataPass

__all__ = [
    'DoWhileLoopPass',
    'ForEachBlockPass',
    'IfThenElsePass',
    'PassPredicate',
    'ChangePredicate',
    'GateCountPredicate',
    'NotPredicate',
    'WhileLoopPass',
    'ClusteringPartitioner',
    'GreedyPartitioner',
    'ScanPartitioner',
    'QuickPartitioner',
    'SynthesisPass',
    'LEAPSynthesisPass',
    'QSearchSynthesisPass',
    'QFASTDecompositionPass',
    'QPredictDecompositionPass',
    'CompressPass',
    'RecordStatsPass',
    'SetRandomSeedPass',
    'UnfoldPass',
    'UpdateDataPass',
    'ToU3Pass',
    'SimpleLayoutPass',
    'ScanningGateRemovalPass',
    'WindowOptimizationPass',
    'SimpleLayerGenerator',
    'AStarHeuristic',
    'GreedyHeuristic',
    'DijkstraHeuristic',
    'Frontier',
    'LayerGenerator',
    'HeuristicFunction',
    'SeedLayerGenerator',
    'BlockConversionPass',
    'StairLayerGenerator',
    'DoThenDecide',
    'SubstitutePass',
    'ParallelDo',
]
