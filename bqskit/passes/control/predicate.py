"""This module defines the PassPredicate base class."""
from __future__ import annotations

import abc
from typing import Any

from bqskit.ir.circuit import Circuit


class PassPredicate(abc.ABC):
    """
    The PassPredicate abstract base class.

    A PassPredicate implements the :func:`get_truth_value` method, which is
    called from control passes to determine the flow of execution.
    """

    @abc.abstractmethod
    def get_truth_value(self, circuit: Circuit, data: dict[str, Any]) -> bool:
        """Call this predicate and retrieve the truth value result."""

    def __call__(self, circuit: Circuit, data: dict[str, Any]) -> bool:
        """Call this predicate and retrieve the truth value result."""

        if not isinstance(circuit, Circuit):
            raise TypeError('Expected Circuit, got %s.' % type(circuit))

        if not isinstance(data, dict):
            raise TypeError('Expected dictionary, got %s.' % type(data))

        return self.get_truth_value(circuit, data)
