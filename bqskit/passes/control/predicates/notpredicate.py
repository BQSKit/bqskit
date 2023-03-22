"""This module implements the NotPredicate class."""
from __future__ import annotations

from typing import TYPE_CHECKING

from bqskit.passes.control.predicate import PassPredicate

if TYPE_CHECKING:
    from bqskit.compiler.passdata import PassData
    from bqskit.ir.circuit import Circuit


class NotPredicate(PassPredicate):
    """
    The NotPredicate class.

    The NotPredicate takes a predicate and always returns the opposite truth
    value.
    """

    def __init__(self, predicate: PassPredicate) -> None:
        """
        Construct a NotPredicate.

        Args:
            predicate (PassPredicate): The predicate to invert.
        """

        if not isinstance(predicate, PassPredicate):
            raise TypeError(f'Expected PassPredicate, got {type(predicate)}.')

        self.predicate = predicate

    def get_truth_value(self, circuit: Circuit, data: PassData) -> bool:
        """Call this predicate, see :class:`PassPredicate` for more info."""
        return not self.predicate(circuit, data)
