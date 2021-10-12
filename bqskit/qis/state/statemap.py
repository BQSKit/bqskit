"""This module implements the StateVectorMap base class."""
from __future__ import annotations

import abc

from bqskit.qis.state.state import StateLike
from bqskit.qis.state.state import StateVector


class StateVectorMap(abc.ABC):
    """A map from quantum pure states to quantum pure states."""

    @abc.abstractmethod
    def get_statevector(self, in_state: StateLike) -> StateVector:
        """
        Calculate the output state given the input state.

        Args:
            in_state (StateLike): The pure quantum state input to the
                state vector map.

        Returns:
            StateVector: The output quantum state.
        """
