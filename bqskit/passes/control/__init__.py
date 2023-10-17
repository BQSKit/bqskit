"""This package defines passes and objects that control pass execution flow."""
from __future__ import annotations

from bqskit.passes.control.dothendecide import DoThenDecide
from bqskit.passes.control.dowhileloop import DoWhileLoopPass
from bqskit.passes.control.foreach import ClearAllBlockData
from bqskit.passes.control.foreach import ForEachBlockPass
from bqskit.passes.control.ifthenelse import IfThenElsePass
from bqskit.passes.control.paralleldo import ParallelDo
from bqskit.passes.control.predicate import PassPredicate
from bqskit.passes.control.predicates.change import ChangePredicate
from bqskit.passes.control.predicates.count import GateCountPredicate
from bqskit.passes.control.predicates.many import ManyQuditGatesPredicate
from bqskit.passes.control.predicates.multi import MultiPhysicalPredicate
from bqskit.passes.control.predicates.notpredicate import NotPredicate
from bqskit.passes.control.predicates.physical import PhysicalPredicate
from bqskit.passes.control.predicates.single import AllConstantSingleQuditGates
from bqskit.passes.control.predicates.single import HasGeneralSingleQuditGate
from bqskit.passes.control.predicates.single import NoSingleQuditGatesInModel
from bqskit.passes.control.predicates.single import SinglePhysicalPredicate
from bqskit.passes.control.predicates.single import ZXGatePredicate
from bqskit.passes.control.predicates.width import WidthPredicate
from bqskit.passes.control.whileloop import WhileLoopPass

__all__ = [
    'DoWhileLoopPass',
    'ForEachBlockPass',
    'ClearAllBlockData',
    'IfThenElsePass',
    'PassPredicate',
    'ChangePredicate',
    'GateCountPredicate',
    'ManyQuditGatesPredicate',
    'NotPredicate',
    'WhileLoopPass',
    'DoThenDecide',
    'ParallelDo',
    'WidthPredicate',
    'PhysicalPredicate',
    'MultiPhysicalPredicate',
    'SinglePhysicalPredicate',
    'NoSingleQuditGatesInModel',
    'HasGeneralSingleQuditGate',
    'ZXGatePredicate',
    'AllConstantSingleQuditGates',
]
