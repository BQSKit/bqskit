"""This module implements the HilbertSchmidtCost and
HilbertSchmidtCostGenerator."""
from __future__ import annotations

from typing import TYPE_CHECKING

from bqskitrs import HilbertSchmidtCostFunction

from bqskit.ir.opt.cost.differentiable import DifferentiableCostFunction
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.qis.state.state import StateVector

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit
    from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
    from bqskit.ir.opt.cost.function import CostFunction


class HilbertSchmidtCost(
    HilbertSchmidtCostFunction,
    DifferentiableCostFunction,
):
    """
    The HilbertSchmidtCost CostFunction implementation.

    The Hilbert-Schmidt CostFuction is a differentiable map from circuit
    parameters to a cost value that is based on the Hilbert-Schmidt inner
    product. This function is global-phase-aware, meaning that the cost is zero
    if the target and circuit unitary differ only by a global phase.
    """


class HilbertSchmidtCostGenerator(CostFunctionGenerator):
    """
    The HilbertSchmidtCostGenerator class.

    This generator produces configured HilbertSchmidtCost functions.
    """

    def gen_cost(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector,
    ) -> CostFunction:
        """Generate a CostFunction, see CostFunctionGenerator for more info."""
        return HilbertSchmidtCost(circuit, target)
