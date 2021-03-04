"""This module implements the StateVectorMap base class."""

import abc

class StateVectorMap(abc.ABC):
    """The StateVectorMap base class."""

    @abc.abstractmethod
    def get_statevector(self, in_state: StateVector) -> StateVector:
        """Calculate the output state given the input state."""
        