"""
This module implements the PassBase class.

All bqskit passes must inherit from the PaseBase class and implement
the run function. A pass represents a sweep over a circuit.
"""
import abc
from typing import Any
from typing import Dict

from bqskit.ir.circuit import Circuit


class PassBase (abc.ABC):
    """The PaseBase Class."""

    def name(self) -> str:
        """Get the name of the pass."""
        return self.__class__.__name__

    @abc.abstractmethod
    def run(self, circuit: Circuit, data: Dict[str, Any] = {}) -> None:
        """
        The run function performs this pass's operation on the circuit.

        Args:
            circuit (Circuit): The circuit to operate on.

            data (Dict[str, Any]): Associated data for the pass.
                Can be used to provide auxillary information from
                previous runs. This function should never fail based on
                what is in this dictionary.
        """
        pass
