"""This module implements the HeuristicFunction base class."""
from __future__ import annotations

import abc

from bqskit.ir.circuit import Circuit
from bqskit.qis.state.state import StateVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class HeuristicFunction(abc.ABC):
    """
    The HeuristicFunction base class.

    A HeuristicFunction is a map from a circuit to a real value.
    """

    @abc.abstractmethod
    def get_value(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector,
    ) -> float:
        """Return the heuristic's value given `circuit` and `target`."""

    def __call__(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector,
    ) -> float:
        """Return the heuristic's value given `circuit` and `target`."""

        if not isinstance(circuit, Circuit):
            raise TypeError(
                'Expected circuit, got %s.' % type(circuit),
            )

        if not isinstance(target, (UnitaryMatrix, StateVector)):
            raise TypeError(
                'Expected unitary or state, got %s.' % type(target),
            )

        return self.get_value(circuit, target)
