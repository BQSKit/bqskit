"""This module implements the DiagonalPredicate class."""
from __future__ import annotations

from typing import TYPE_CHECKING

from bqskit.passes.control.predicate import PassPredicate
from bqskit.utils.math import diagonal_distance

if TYPE_CHECKING:
    from bqskit.compiler.passdata import PassData
    from bqskit.ir.circuit import Circuit


class DiagonalPredicate(PassPredicate):
    """
    The DiagonalPredicate class.

    The DiagonalPredicate class returns True if the circuit's unitary can
    be approximately inverted by a diagonal unitary. A unitary is approx-
    imately inverted when the Hilbert-Schmidt distance to the identity is
    less than some threshold.
    """

    def __init__(self, threshold: float) -> None:
        """
        Construct a DiagonalPredicate.

        Args:
            threshold (float): If a circuit can be approximately inverted
                by a diagonal unitary (meaning the Hilbert-Schmidt distance
                to the identity is less than or equal to this number after
                multiplying by the diagonal unitary), True is returned.
        """
        self.threshold = threshold

    def get_truth_value(self, circuit: Circuit, data: PassData) -> bool:
        """Call this predicate, see :class:`PassPredicate` for more info."""
        dist = diagonal_distance(circuit.get_unitary())
        return dist <= self.threshold
