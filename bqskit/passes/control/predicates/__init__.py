"""This package implements predicates for use in control passes."""
from __future__ import annotations

from bqskit.passes.control.predicates.change import ChangePredicate
from bqskit.passes.control.predicates.count import GateCountPredicate
from bqskit.passes.control.predicates.notpredicate import NotPredicate

__all__ = ['ChangePredicate', 'GateCountPredicate', 'NotPredicate']
