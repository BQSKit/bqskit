"""This package defines passes and objects that control pass execution flow."""
from __future__ import annotations

from bqskit.compiler.passes.control.predicate import PassPredicate
from bqskit.compiler.passes.control.predicates.change import ChangePredicate

__all__ = ['PassPredicate', 'ChangePredicate']
