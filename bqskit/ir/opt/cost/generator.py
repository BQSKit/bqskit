"""This module implements the CostFunctionGenerator base classes."""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bqskit.ir.opt.cost import CostFunction
    from bqskit.ir.circuit import Circuit
    from bqskit.qis.state.state import StateVector
    from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class CostFunctionGenerator(abc.ABC):
    """
    The CostFunctionGenerator base class.

    A CostFunctionGenerator in BQSKit is a function generator that can
    produce maps from circuit parameters to real numbers.

    The `gen_cost` method signature includes a circuit and target unitary
    or state as parameters. This allows a user to configure the generator
    and pass it to anything that does instantiation, like a synthesis pass,
    which in turn, will generate configured CostFunctions. This is useful
    since passes might be working with changing circuits and will need to
    regenerate CostFunctions from time to time.
    """

    @abc.abstractmethod
    def gen_cost(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector,
    ) -> CostFunction:
        """
        Generate a function from a circuit and target that maps params to cost.

        Args:
            circuit (Circuit): The circuit the cost function is generated for.

            target (UnitaryMatrix | StateVector): The target object.

        Returns:
            (CostFunction): The primitive cost function
                that can be directly passed to a minimizer. This maps
                parameters or circuit inputs to a cost.
        """

    def calc_cost(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector,
    ) -> float:
        """Generate and calculate the cost from the CostFunction."""
        return self.gen_cost(circuit, target).get_cost(circuit.params)

    def __call__(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector,
    ) -> float:
        """Generate and calculate the cost from the CostFunction."""
        return self.calc_cost(circuit, target)
