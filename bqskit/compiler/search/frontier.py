"""This module implements the Frontier class."""
from __future__ import annotations

import heapq
import itertools

from bqskit.compiler.search.heuristic import HeuristicFunction
from bqskit.ir.circuit import Circuit
from bqskit.qis.state.state import StateVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class Frontier:
    """The Frontier class."""

    def __init__(
        self,
        target: UnitaryMatrix | StateVector,
        heuristic_function: HeuristicFunction,
    ) -> None:
        """
        Construct an empty frontier.

        Args:
            target (UnitaryMatrix | StateVector): The target to pass to
                the heuristic_function.

            heuristic_function (HeuristicFunction): The heuristic used
                to sort the Frontier.
        """

        if not isinstance(target, (UnitaryMatrix, StateVector)):
            raise TypeError(
                'Expected unitary or state, got %s.' % type(target),
            )

        if not isinstance(heuristic_function, HeuristicFunction):
            raise TypeError(
                'Expected HeursiticFunction, got %s.'
                % type(heuristic_function),
            )

        self.target = target
        self.heuristic_function = heuristic_function
        self._frontier: list[tuple[float, int, Circuit]] = []
        self._counter = itertools.count()

    def add(self, circuit: Circuit) -> None:
        """Add `circuit` into the frontier."""
        heuristic_value = self.heuristic_function(circuit, self.target)
        count = next(self._counter)
        heapq.heappush(self._frontier, (heuristic_value, count, circuit))

    def pop(self) -> Circuit:
        """Pop the top circuit."""
        return heapq.heappop(self._frontier)[2]

    def empty(self) -> bool:
        """Return true if the frontier is empty."""
        return len(self._frontier) == 0

    def clear(self) -> None:
        """Remove all elements from the frontier."""
        self._frontier.clear()
