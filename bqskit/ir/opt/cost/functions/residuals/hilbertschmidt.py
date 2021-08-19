"""This module implements the HilbertSchmidtCost and HilbertSchmidtGenerator."""
from __future__ import annotations

from typing import TYPE_CHECKING

from bqskitrs import HilbertSchmidtResidualsFunction

from bqskit.ir.opt.cost.differentiable import DifferentiableResidualsFunction
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.qis.state.state import StateVector

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit
    from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
    from bqskit.ir.opt.cost.function import CostFunction


class HilbertSchmidtResiduals(
    HilbertSchmidtResidualsFunction,
    DifferentiableResidualsFunction,
):
    """
    The HilbertSchmidtResiduals CostFunction implementation.

    The Hilbert-Schmidt Residuals CostFuction is a differentiable map from
    circuit parameters to a residual vector that is based on the Hilbert-Schmidt
    inner product. This function is global-phase-aware, meaning that the cost is
    zero if the target and circuit unitary differ only by a global phase.
    """


class HilbertSchmidtResidualsGenerator(CostFunctionGenerator):
    """
    The HilbertSchmidtResidualsGenerator class.

    This generator produces configured HilbertSchmidtResiduals functions.
    """

    def gen_cost(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector,
    ) -> CostFunction:
        """Generate a CostFunction, see CostFunctionGenerator for more info."""
        return HilbertSchmidtResiduals(circuit, target)
