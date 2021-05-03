"""This module implements the HSDistance CostFunction."""
from __future__ import annotations
from bqskit.ir.opt.costgenerator import CostFunctionGenerator

from typing import Sequence, TYPE_CHECKING

import numpy as np

from bqskit.qis.state.state import StateVector
from bqskit.ir.opt.costfunction import CostFunction

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit
    from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class HSDistanceFunction(CostFunction):

    def __init__(
        self, 
        circuit: Circuit,
        target: UnitaryMatrix | StateVector,
    ) -> None:
        self.target = target
        self.circuit = circuit
        self.target_h = target.get_dagger().get_numpy()
        self.dem = target.get_dim()
        
    def get_cost(self, params: Sequence[float]) -> float:
        """Return the cost value given the input parameters."""
        utry = self.circuit.get_unitary(params).get_numpy()
        num = np.abs(np.trace(self.target_h @ utry))
        return 1 - (num / self.dem)

    def get_grad(self, params: Sequence[float]) -> np.ndarray:
        """Return the cost gradient given the input parameters."""
        M, dM = self.circuit.get_unitary_and_grad(params)
        trace_prod = np.trace(self.target_h @ M)
        num = np.abs(trace_prod)
        d_trace_prod = np.array([np.trace(self.target_h @ pM) for pM in dM])
        jacs = -(
            np.real(trace_prod) * np.real(d_trace_prod)
            + np.imag(trace_prod) * np.imag(d_trace_prod)
        )
        jacs *= self.dem / num
        return jacs

    def get_cost_and_grad(
        self,
        params: Sequence[float]
    ) -> tuple[float, np.ndarray]:
        """Return the cost and gradient given the input parameters."""
        M, dM = self.circuit.get_unitary_and_grad(params)
        trace_prod = np.trace(self.target_h @ M.get_numpy())
        num = np.abs(trace_prod)
        obj = 1 - (num / self.dem)
        d_trace_prod = np.array([np.trace(self.target_h @ pM) for pM in dM])
        jacs = -(
            np.real(trace_prod) * np.real(d_trace_prod)
            + np.imag(trace_prod) * np.imag(d_trace_prod)
        )
        jacs *= self.dem / num
        return obj, jacs

class HSDistance(CostFunctionGenerator):
    """
    The HSDistance class.

    This is an CostFunction based on the hilbert-schmidt inner-product.
    """

    def gen_cost(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector,
    ) -> CostFunction:
        """Generate the CostFunction, see CostFunctionGenerator for more info."""
        return HSDistanceFunction(circuit, target)
