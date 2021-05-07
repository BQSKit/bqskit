"""This module implements the Instantiater base class."""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit
    from bqskit.qis.state.state import StateLike
    from bqskit.qis.unitary.unitarymatrix import UnitaryLike


class Instantiater(abc.ABC):
    """
    The Instantiater class.

    An Instantiater is responsible for instantiating circuit templates
    such that the resulting circuit bests implements the desired target.
    """

    @abc.abstractmethod
    def instantiate(
        self,
        circuit: Circuit,
        target: UnitaryLike | StateLike,
    ) -> np.ndarray:
        """
        Instantiate `circuit` to best implement `target`.

        Args:
            circuit (Circuit): The circuit template to instantiate.

            target (UnitaryLike | StateLike): The unitary matrix or
                state vector to implement.

        Returns:
            (list[float]): The list of paremeters for the circuit that
                makes the circuit best implement `target`.

        Notes:
            This method should be side-effect free. This is necessary since
            many instantiate calls to the same circuit using the same
            Instantiater object may happen in parallel.
        """

    @staticmethod
    @abc.abstractmethod
    def is_capable(circuit: Circuit) -> bool:
        """Return true if the circuit can be instantiated."""

    @staticmethod
    @abc.abstractmethod
    def get_violation_report(circuit: Circuit) -> str:
        """Return a message explaining why `circuit` cannot be instantiated."""
