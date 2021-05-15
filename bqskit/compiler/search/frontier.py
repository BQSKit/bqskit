"""This module implements the Frontier class."""
from __future__ import annotations

from bqskit.compiler.search.heuristic import HeuristicFunction
from bqskit.ir.circuit import Circuit


class Frontier:
    """The Frontier class."""

    def __init__(self, heuristic_function: HeuristicFunction) -> None:
        """
        Construct an empty frontier.

        Args:
            heuristic_function (HeuristicFunction): The heuristic used
                to sort the Frontier.
        """

        if not isinstance(heuristic_function, HeuristicFunction):
            raise TypeError(
                'Expected HeursiticFunction, got %s.'
                % type(heuristic_function),
            )

        self.heuristic_function = heuristic_function

    def add(self, circuit: Circuit) -> None:
        """Add `circuit` into the frontier."""
        pass  # TODO

    def pop(self) -> Circuit:
        """Pop the top circuit."""
        pass  # TODO

    def empty(self) -> bool:
        """Return true if the frontier is empty."""
        return True
