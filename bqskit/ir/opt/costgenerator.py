"""This module implements the CostFunctionGenerator base classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import abc
from bqskit.ir.opt.costfunction import CostFunction

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit
    from bqskit.qis.state.state import StateVector
    from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class CostFunctionGenerator(abc.ABC):
    """
    The CostFunctionGenerator base class.

    A CostFunctionGenerator in BQSKit is a differentiable function
    generator that can produce maps from circuit parameters to real numbers
    and their derivatives.

    When subclassing, you will need to implement the gen_cost and gen_grad
    function factories. You can overwrite gen_cost_and_grad for optimization
    in some cases.
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

    # @abc.abstractmethod
    # def gen_grad(
    #     self,
    #     circuit: Circuit,
    #     target: UnitaryMatrix | StateVector,
    # ) -> Callable[[Sequence[float], np.ndarray]]:
    #     """
    #     Generate a map from params to the gradient of the cost.

    #     Args:
    #         circuit (Circuit): The circuit the cost function is generated for.

    #         target (UnitaryMatrix | StateVector): The target object.

    #     Returns:
    #         (Callable[[Sequence[float], np.ndarray]]): The primitive cost
    #             gradient function that can be directly passed to a minimizer.
    #             This maps parameters or circuit inputs to their derivatives.
    #     """

    # def gen_cost_and_grad(
    #     self,
    #     circuit: Circuit,
    #     target: UnitaryMatrix | StateVector,
    # ) -> Callable[[Sequence[float]], tuple[float, np.ndarray]]:
    #     """Generate a map from params to the cost and gradient of the cost."""
    #     cost_fn = self.gen_cost(circuit, target)
    #     grad_fn = self.gen_grad(circuit, target)

    #     def cost_and_grad(params: Sequence[float]) -> tuple[float, np.ndarray]:
    #         return cost_fn(params), grad_fn(params)
    #     return cost_and_grad

    # def calc_cost(
    #     self,
    #     circuit: Circuit,
    #     target: UnitaryMatrix | StateVector,
    # ) -> float:
    #     """Return the cost given a circuit and target."""
    #     return self.gen_cost(circuit, target).get_cost(circuit.get_params())

    # def calc_grad(
    #     self,
    #     circuit: Circuit,
    #     target: UnitaryMatrix | StateVector,
    # ) -> np.ndarray:
    #     """Return the gradient given a circuit and target."""
    #     return self.gen_grad(circuit, target)(circuit.get_params())

    # def calc_cost_and_grad(
    #     self,
    #     circuit: Circuit,
    #     target: UnitaryMatrix | StateVector,
    # ) -> np.ndarray:
    #     """Return the cost and gradient given a circuit and target."""
    #     return self.gen_cost_and_grad(circuit, target)(circuit.get_params())
