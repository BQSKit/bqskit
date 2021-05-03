"""This module implements the Minimizer base class."""
from __future__ import annotations

import abc

import numpy as np
from bqskit.ir.opt.costfunction import CostFunction

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit


class Minimizer(abc.ABC):
    """
    The Minimizer class.

    An minimizer finds the parameters for a circuit template that minimizes
    some CostFunction.
    """

    @abc.abstractmethod
    def minimize(self, circuit: Circuit, cost: CostFunction) -> np.ndarray:
        """
        Minimize the circuit with respect to some cost function.

        Args:
            circuit (Circuit): The circuit whose parameters are being explored.

            cost (CostFunction): The CostFunction to minimize.
        
        Returns:
            (np.ndarray): The circuit parameters that minimizes the cost.
        
        Notes:
            This function should be side-effect free. This is because many
            calls may be running in parallel on the same circuit.
        """
