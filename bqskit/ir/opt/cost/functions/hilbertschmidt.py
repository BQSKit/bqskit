"""This module implements the HilbertSchmidtCost and HilbertSchmidtGenerator."""
from __future__ import annotations

from typing import Sequence
from typing import TYPE_CHECKING

import numpy as np

from bqskit.ir.opt.cost.differentiable import DifferentiableCostFunction
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.qis.state.state import StateVector

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit
    from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
    from bqskit.ir.opt.cost.function import CostFunction


class HilbertSchmidtCost(DifferentiableCostFunction):
    """
    The HilbertSchmidtCost CostFunction implementation.

    The Hilbert-Schmidt CostFuction is a differentiable map from circuit
    parameters to a cost value that is based on the Hilbert-Schmidt inner
    product. This function is global-phase-aware, meaning that the cost is zero
    if the target and circuit unitary differ only by a global phase.

    """

    def __init__(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector,
    ) -> None:
        """Construct a HilbertSchmidtCost function."""
        self.target = target
        self.circuit = circuit
        self.target_h = target.get_dagger().get_numpy()
        self.dem = target.get_dim()

    def get_cost(self, params: Sequence[float]) -> float:
        """Return the cost value given the input parameters."""
        utry = self.circuit.get_unitary(params).get_numpy()
        num = np.abs(np.sum(self.target_h * utry))
        return 1 - (num / self.dem)

    def get_grad(self, params: Sequence[float]) -> np.ndarray:
        """Return the cost gradient given the input parameters."""
        return self.get_cost_and_grad(params)[1]

    def get_cost_and_grad(
        self,
        params: Sequence[float],
    ) -> tuple[float, np.ndarray]:
        """Return the cost and gradient given the input parameters."""
        M, dM = self.circuit.get_unitary_and_grad(params)
        trace_prod = np.sum(self.target_h * M.get_numpy())
        num = np.abs(trace_prod)
        cost = 1 - (num / self.dem)
        d_trace_prod = np.array([np.sum(self.target_h * pM) for pM in dM])
        grad = -(
            np.real(trace_prod) * np.real(d_trace_prod)
            + np.imag(trace_prod) * np.imag(d_trace_prod)
        )
        grad *= self.dem / num
        return cost, grad


class HilbertSchmidtGenerator(CostFunctionGenerator):
    """
    The HilbertSchmidtGenerator class.

    This generator produces configured HilbertSchmidtCost functions.

    """

    def gen_cost(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector,
    ) -> CostFunction:
        """Generate a CostFunction, see CostFunctionGenerator for more info."""
        return HilbertSchmidtCost(circuit, target)
