"""This module implements the AStarHeuristic class."""
from __future__ import annotations

from bqskit.ir.circuit import Circuit
from bqskit.ir.opt.cost import CostFunctionGenerator
from bqskit.ir.opt.cost import HilbertSchmidtCostGenerator
from bqskit.passes.search.heuristic import HeuristicFunction
from bqskit.qis.state.state import StateVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_real_number


class AStarHeuristic(HeuristicFunction):
    """
    The AStarHeuristic HeuristicFunction class.

    Defines a heuristic that combines the depth of the circuit with its cost.
    It generally gives similar quality results to DjikstraHeuristic, but with
    a drastic reduction in the number of instantiation calls.

    In pure search terms: f(p) = cost(p) + heuristic(p), this implements
    the f(p) component.
    """

    def __init__(
        self,
        heuristic_factor: float = 10.0,
        cost_factor: float = 1.0,
        cost_gen: CostFunctionGenerator = HilbertSchmidtCostGenerator(),
    ) -> None:
        """
        Construct a AStarHeuristic Function.

        Args:
            heuristic_factor (float): Scale the heuristic component by
                this value.

            cost_factor (float): Scale the cost component by this value.

            cost_gen (CostFunctionGenerator): This is used to generate
                cost functions used during evaluations.
        """
        if not is_real_number(heuristic_factor):
            raise TypeError(
                'Expected float for heuristic_factor, got %s.'
                % type(heuristic_factor),
            )

        if not is_real_number(cost_factor):
            raise TypeError(
                'Expected float for cost_factor, got %s.'
                % type(cost_factor),
            )

        if not isinstance(cost_gen, CostFunctionGenerator):
            raise TypeError(
                'Expected CostFunctionGenerator for cost_gen, got %s.'
                % type(cost_gen),
            )

        self.heuristic_factor = heuristic_factor
        self.cost_factor = cost_factor
        self.cost_gen = cost_gen

    def get_value(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector,
    ) -> float:
        """Return the heuristic's value, see HeuristicFunction for more info."""
        cost = 0.0
        for gate in circuit.gate_set:
            if gate.num_qudits == 1:
                continue
            cost += float(circuit.count(gate))

        heuristic = self.cost_gen.calc_cost(circuit, target)
        return self.heuristic_factor * heuristic + self.cost_factor * cost
