"""This module implements the Optimizer base class."""

import abc
from bqskit.ir.circuit import Circuit

class Optimizer(abc.ABC):
    """
    The Optimizer class.
    """

    @abc.abstractmethod
    def optimize(circuit: Circuit) -> None:
        """
        Optimize the circuit.
        """