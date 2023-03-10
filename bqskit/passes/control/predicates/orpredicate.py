"""This module implements the OrPredicate class."""
from __future__ import annotations

from typing import TYPE_CHECKING

from bqskit.passes.control.predicate import PassPredicate

if TYPE_CHECKING:
    from bqskit.compiler.passdata import PassData
    from bqskit.ir.circuit import Circuit


class OrPredicate(PassPredicate):
    """
    The OrPredicate class.

    The OrPredicate takes two predicates and returns true if either are true.
    """

    def __init__(self, p1: PassPredicate, p2: PassPredicate) -> None:
        """
        Construct a OrPredicate.

        Args:
            p1 (PassPredicate): The first predicate.

            p2 (PassPredicate): The second predicate.
        """

        if not isinstance(p1, PassPredicate):
            raise TypeError(f'Expected PassPredicate, got {type(p1)}.')

        if not isinstance(p2, PassPredicate):
            raise TypeError(f'Expected PassPredicate, got {type(p2)}.')

        self.p1 = p1
        self.p2 = p2

    def get_truth_value(self, circuit: Circuit, data: PassData) -> bool:
        """Call this predicate, see :class:`PassPredicate` for more info."""
        return self.p1(circuit, data) or self.p2(circuit, data)
