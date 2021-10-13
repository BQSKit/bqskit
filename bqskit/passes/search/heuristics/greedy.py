"""This module implements the GreedyHeuristic class."""
from __future__ import annotations

from bqskit.ir.circuit import Circuit
from bqskit.ir.opt.cost import CostFunctionGenerator
from bqskit.ir.opt.cost import HilbertSchmidtCostGenerator
from bqskit.passes.search.heuristic import HeuristicFunction
from bqskit.qis.state.state import StateVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class GreedyHeuristic(HeuristicFunction):
    """
    The GreedyHeuristic HeuristicFunction class.

    Defines a heuristic that results in greedy search. This function only looks
    at the current distance of the circuit from the target. This will create a
    behavior similar to depth-first search.
    """

    def __init__(
        self,
        cost_gen: CostFunctionGenerator = HilbertSchmidtCostGenerator(),
    ) -> None:
        """
        Construct a GreedyHeuristic Function.

        Args:
            cost_gen (CostFunctionGenerator): This is used to generate
                cost functions used during evaluations.
        """
        if not isinstance(cost_gen, CostFunctionGenerator):
            raise TypeError(
                'Expected CostFunctionGenerator for cost_gen, got %s.'
                % type(cost_gen),
            )

        self.cost_gen = cost_gen

    def get_value(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector,
    ) -> float:
        """Return the heuristic's value, see HeuristicFunction for more info."""
        return self.cost_gen.calc_cost(circuit, target)
