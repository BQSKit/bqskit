"""This package implements predicates for use in control passes."""
from __future__ import annotations

from bqskit.passes.control.predicates.andpredicate import AndPredicate
from bqskit.passes.control.predicates.change import ChangePredicate
from bqskit.passes.control.predicates.count import GateCountPredicate
from bqskit.passes.control.predicates.multi import MultiPhysicalPredicate
from bqskit.passes.control.predicates.notpredicate import NotPredicate
from bqskit.passes.control.predicates.orpredicate import OrPredicate
from bqskit.passes.control.predicates.physical import PhysicalPredicate
from bqskit.passes.control.predicates.single import SinglePhysicalPredicate
from bqskit.passes.control.predicates.width import WidthPredicate

__all__ = [
    'ChangePredicate',
    'GateCountPredicate',
    'AndPredicate',
    'NotPredicate',
    'OrPredicate',
    'WidthPredicate',
    'PhysicalPredicate',
    'SinglePhysicalPredicate',
    'MultiPhysicalPredicate',
]
