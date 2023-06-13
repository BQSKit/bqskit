"""This module defines the PassPredicate base class."""
from __future__ import annotations

import abc

from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit


class PassPredicate(abc.ABC):
    """
    The PassPredicate abstract base class.

    A PassPredicate implements the :func:`get_truth_value` method, which is
    called from control passes to determine the flow of execution.
    """

    @abc.abstractmethod
    def get_truth_value(self, circuit: Circuit, data: PassData) -> bool:
        """Call this predicate and retrieve the truth value result."""

    def __call__(self, circuit: Circuit, data: PassData) -> bool:
        """Call this predicate and retrieve the truth value result."""

        if not isinstance(circuit, Circuit):
            raise TypeError(f'Expected Circuit, got {type(circuit)}.')

        if not isinstance(data, PassData):
            raise TypeError(f'Expected PassData, got {type(data)}.')

        return self.get_truth_value(circuit, data)
