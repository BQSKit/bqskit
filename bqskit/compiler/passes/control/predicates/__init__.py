"""This package implements predicates for use in control passes."""
from __future__ import annotations

from bqskit.compiler.passes.control.predicates.change import ChangePredicate
from bqskit.compiler.passes.control.predicates.count import GateCountPredicate
from bqskit.compiler.passes.control.predicates.notpredicate import NotPredicate

__all__ = ['ChangePredicate', 'GateCountPredicate', 'NotPredicate']
