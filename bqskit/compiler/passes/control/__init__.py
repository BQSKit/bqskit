# type: ignore
# TODO: Remove type: ignore, when new mypy comes out with TypeGuards
"""This package defines passes and objects that control pass execution flow."""
from __future__ import annotations

from bqskit.compiler.passes.control.dowhileloop import DoWhileLoopPass
from bqskit.compiler.passes.control.ifthenelse import IfThenElsePass
from bqskit.compiler.passes.control.predicate import PassPredicate
from bqskit.compiler.passes.control.predicates.change import ChangePredicate
from bqskit.compiler.passes.control.whileloop import WhileLoopPass

__all__ = [
    'DoWhileLoopPass',
    'IfThenElsePass',
    'PassPredicate',
    'ChangePredicate',
    'WhileLoopPass',
]
