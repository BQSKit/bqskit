"""
This module implements the BasePass class.

All bqskit passes must inherit from the BasePass class and implement the
run function. A pass represents an operation on a circuit.
"""
from __future__ import annotations

import abc
from typing import Any

from bqskit.ir.circuit import Circuit


class BasePass (abc.ABC):
    """The BasePass Class."""

    def name(self) -> str:
        """Get the name of the pass."""
        return self.__class__.__name__

    @abc.abstractmethod
    def run(self, circuit: Circuit, data: dict[str, Any]) -> None:
        """
        The run function performs this pass's operation on the circuit.

        Args:
            circuit (Circuit): The circuit to operate on.

            data (Dict[str, Any]): Associated data for the pass.
                Can be used to provide auxillary information from
                previous runs. This function should never fail based on
                what is in this dictionary.
        """

    def __getstate__(self) -> Any:
        raise NotImplementedError()  # TODO

    def __setstate__(self, state: Any) -> None:
        raise NotImplementedError()  # TODO
