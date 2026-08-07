"""This module implements the PhysicalPredicate class."""
from __future__ import annotations

from typing import TYPE_CHECKING

from bqskit.passes.control.predicate import PassPredicate

if TYPE_CHECKING:
    from bqskit.compiler.passdata import PassData
    from bqskit.ir.circuit import Circuit


class PhysicalPredicate(PassPredicate):
    """
    The PhysicalPredicate class.

    The PhysicalPredicate returns true if the circuit can be executed on the
    workflow's machine model with the current placement.
    """

    def get_truth_value(self, circuit: Circuit, data: PassData) -> bool:
        """Call this predicate, see :class:`PassPredicate` for more info."""
        return data.model.is_compatible(circuit, data.placement)
