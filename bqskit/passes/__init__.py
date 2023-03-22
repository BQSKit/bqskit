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
    GroupSingleQuditGatePass

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
    SynthesisPass

.. rubric:: Processing Passes

.. autosummary::
    :toctree: autogen
    :recursive:

    ExhaustiveGateRemovalPass
    IterativeScanningGateRemovalPass
    Rebase2QuditGatePass
    ScanningGateRemovalPass
    SubstitutePass

.. rubric:: Control Passes

This are passes that are used to manage control flow inside of a workflow.

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
    WidthPredicate
    PhysicalPredicate
    SinglePhysicalPredicate
    MultiPhysicalPredicate

.. rubric:: Rule-based Passes

These passes apply fixed rules to a circuit to transform it.

.. autosummary::
    :toctree: autogen
    :recursive:

    CHToCNOTPass
    CNOTToCZPass
    CNOTToCHPass
    CNOTToCYPass
    CYToCNOTPass
    SwapToCNOTPass
    U3Decomposition
    ZXZXZDecomposition

.. rubric:: Mapping Passes

These passes either perform qubit placement, layout, routing or otherwise
are involved the qubit mapping process.

.. autosummary::
    :toctree: autogen
    :recursive:

    GeneralizedSabreLayoutPass
    GreedyPlacementPass
    TrivialPlacementPass
    GeneralizedSabreRoutingPass
    SetModelPass
    ApplyPlacement

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
    LogPass
    ExtendBlockSizePass
    LogErrorPass
    FillSingleQuditGatesPass

.. rubric:: IO Passes

.. autosummary::
    :toctree: autogen
    :recursive:

    LoadCheckpointPass
    SaveCheckpointPass
    SaveIntermediatePass
    RestoreIntermediatePass

.. rubric:: Search Heuristics

.. autosummary::
    :toctree: autogen
    :recursive:

    HeuristicFunction
    AStarHeuristic
    GreedyHeuristic
    DijkstraHeuristic

.. rubric:: Search Layer Generators

Layer generators can be used to modify how search based synthesis
algorithms extend circuit templates.

.. autosummary::
    :toctree: autogen
    :recursive:

    FourParamGenerator
    MiddleOutLayerGenerator
    SeedLayerGenerator
    SimpleLayerGenerator
    SingleQuditLayerGenerator
    StairLayerGenerator
    WideLayerGenerator
"""
from __future__ import annotations

from bqskit.passes.alias import PassAlias
from bqskit.passes.control.dothendecide import DoThenDecide
from bqskit.passes.control.dowhileloop import DoWhileLoopPass
from bqskit.passes.control.foreach import ForEachBlockPass
from bqskit.passes.control.ifthenelse import IfThenElsePass
from bqskit.passes.control.paralleldo import ParallelDo
from bqskit.passes.control.predicate import PassPredicate
from bqskit.passes.control.predicates.change import ChangePredicate
from bqskit.passes.control.predicates.count import GateCountPredicate
from bqskit.passes.control.predicates.multi import MultiPhysicalPredicate
from bqskit.passes.control.predicates.notpredicate import NotPredicate
from bqskit.passes.control.predicates.physical import PhysicalPredicate
from bqskit.passes.control.predicates.single import SinglePhysicalPredicate
from bqskit.passes.control.predicates.width import WidthPredicate
from bqskit.passes.control.whileloop import WhileLoopPass
from bqskit.passes.group import PassGroup
from bqskit.passes.io.checkpoint import LoadCheckpointPass
from bqskit.passes.io.checkpoint import SaveCheckpointPass
from bqskit.passes.io.intermediate import RestoreIntermediatePass
from bqskit.passes.io.intermediate import SaveIntermediatePass
from bqskit.passes.mapping.apply import ApplyPlacement
from bqskit.passes.mapping.layout.sabre import GeneralizedSabreLayoutPass
from bqskit.passes.mapping.placement.greedy import GreedyPlacementPass
from bqskit.passes.mapping.placement.trivial import TrivialPlacementPass
from bqskit.passes.mapping.routing.sabre import GeneralizedSabreRoutingPass
from bqskit.passes.mapping.setmodel import SetModelPass
from bqskit.passes.measure import ExtractMeasurements
from bqskit.passes.measure import RestoreMeasurements
from bqskit.passes.noop import NOOPPass
from bqskit.passes.partitioning.cluster import ClusteringPartitioner
from bqskit.passes.partitioning.greedy import GreedyPartitioner
from bqskit.passes.partitioning.quick import QuickPartitioner
from bqskit.passes.partitioning.scan import ScanPartitioner
from bqskit.passes.partitioning.single import GroupSingleQuditGatePass
from bqskit.passes.processing.exhaustive import ExhaustiveGateRemovalPass
from bqskit.passes.processing.iterative import IterativeScanningGateRemovalPass
from bqskit.passes.processing.rebase import Rebase2QuditGatePass
from bqskit.passes.processing.scan import ScanningGateRemovalPass
from bqskit.passes.processing.substitute import SubstitutePass
from bqskit.passes.rules.ch2cnot import CHToCNOTPass
from bqskit.passes.rules.cnot2ch import CNOTToCHPass
from bqskit.passes.rules.cnot2cy import CNOTToCYPass
from bqskit.passes.rules.cnot2cz import CNOTToCZPass
from bqskit.passes.rules.cy2cnot import CYToCNOTPass
from bqskit.passes.rules.swap2cnot import SwapToCNOTPass
from bqskit.passes.rules.u3 import U3Decomposition
from bqskit.passes.rules.zxzxz import ZXZXZDecomposition
from bqskit.passes.search.frontier import Frontier
from bqskit.passes.search.generator import LayerGenerator
from bqskit.passes.search.generators.fourparam import FourParamGenerator
from bqskit.passes.search.generators.middleout import MiddleOutLayerGenerator
from bqskit.passes.search.generators.seed import SeedLayerGenerator
from bqskit.passes.search.generators.simple import SimpleLayerGenerator
from bqskit.passes.search.generators.single import SingleQuditLayerGenerator
from bqskit.passes.search.generators.stair import StairLayerGenerator
from bqskit.passes.search.generators.wide import WideLayerGenerator
from bqskit.passes.search.heuristic import HeuristicFunction
from bqskit.passes.search.heuristics.astar import AStarHeuristic
from bqskit.passes.search.heuristics.dijkstra import DijkstraHeuristic
from bqskit.passes.search.heuristics.greedy import GreedyHeuristic
from bqskit.passes.synthesis.leap import LEAPSynthesisPass
from bqskit.passes.synthesis.qfast import QFASTDecompositionPass
from bqskit.passes.synthesis.qpredict import QPredictDecompositionPass
from bqskit.passes.synthesis.qsearch import QSearchSynthesisPass
from bqskit.passes.synthesis.synthesis import SynthesisPass
from bqskit.passes.synthesis.target import SetTargetPass
from bqskit.passes.util.compress import CompressPass
from bqskit.passes.util.conversion import BlockConversionPass
from bqskit.passes.util.converttou3 import ToU3Pass
from bqskit.passes.util.extend import ExtendBlockSizePass
from bqskit.passes.util.fill import FillSingleQuditGatesPass
from bqskit.passes.util.log import LogErrorPass
from bqskit.passes.util.log import LogPass
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
    'ScanningGateRemovalPass',
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
    'LoadCheckpointPass',
    'SaveCheckpointPass',
    'SaveIntermediatePass',
    'RestoreIntermediatePass',
    'GroupSingleQuditGatePass',
    'SingleQuditLayerGenerator',
    'MiddleOutLayerGenerator',
    'CNOTToCZPass',
    'ZXZXZDecomposition',
    'ExhaustiveGateRemovalPass',
    'WidthPredicate',
    'IterativeScanningGateRemovalPass',
    'SetModelPass',
    'PassAlias',
    'PassGroup',
    'NOOPPass',
    'FourParamGenerator',
    'WideLayerGenerator',
    'SwapToCNOTPass',
    'GeneralizedSabreLayoutPass',
    'GreedyPlacementPass',
    'TrivialPlacementPass',
    'GeneralizedSabreRoutingPass',
    'SetModelPass',
    'U3Decomposition',
    'PhysicalPredicate',
    'SinglePhysicalPredicate',
    'Rebase2QuditGatePass',
    'LogPass',
    'ExtractMeasurements',
    'RestoreMeasurements',
    'ExtendBlockSizePass',
    'ApplyPlacement',
    'MultiPhysicalPredicate',
    'LogErrorPass',
    'FillSingleQuditGatesPass',
    'CHToCNOTPass',
    'CNOTToCHPass',
    'CNOTToCYPass',
    'CYToCNOTPass',
    'SetTargetPass',
]
