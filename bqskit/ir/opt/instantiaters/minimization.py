"""This module implements the Minimization class."""
from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt.cost.functions import HilbertSchmidtCostGenerator  
from bqskit.ir.opt.cost.residual import ResidualsFunction  
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.ir.opt.instantiater import Instantiater
from bqskit.ir.opt.minimizer import Minimizer
from bqskit.ir.opt.minimizers.ceres import CeresMinimizer
from bqskit.ir.opt.multistartgens.random import RandomStartGenerator
from bqskit.qis.state.state import StateLike
from bqskit.qis.state.state import StateVector

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit
    from bqskit.qis.state.system import StateSystem
    from bqskit.qis.state.system import StateSystemLike
    from bqskit.qis.unitary.unitarymatrix import UnitaryLike
    from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class Minimization(Instantiater):
    """The Minimization circuit instantiater."""

    def __init__(
        self,
        cost_fn_gen: CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
        minimizer: Minimizer = CeresMinimizer(),
        **kwargs: dict[str, Any],  # TODO: handle dist_tol and other options
    ) -> None:
        """
        Construct and configure a Minimization Instantiater.

        Args:
            cost_fn_gen (CostFunctionGenerator): This generator creates cost
                functions that are minimized.
                (Default: HilbertSchmidtGenerator())

            minimizer (Minimizer): The minimizer to use. If left as
                None, attempts to select best one.
        """

        if not isinstance(cost_fn_gen, CostFunctionGenerator):
            raise TypeError(
                'Expected CostFunctionGenerator, got %s.' % type(cost_fn_gen),
            )

        if not isinstance(minimizer, Minimizer):
            raise TypeError('Expected Minimizer, got %s.' % type(minimizer))

        self.cost_fn_gen = cost_fn_gen
        self.minimizer = minimizer

    def instantiate(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector | StateSystem,
        x0: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Instantiate `circuit`, see Instantiater for more info."""
        cost = self.cost_fn_gen.gen_cost(circuit, target)
        return self.minimizer.minimize(cost, x0)

    @staticmethod
    def is_capable(circuit: Circuit) -> bool:
        """Return true if the circuit can be instantiated."""
        return all(
            not isinstance(g, VariableUnitaryGate)
            for g in circuit.gate_set
        )

    @staticmethod
    def get_violation_report(circuit: Circuit) -> str:
        """
        All circuits can be instantiated with minimization.

        See Instantiater for more info.
        """
        return (
            'Cannot instantiate a circuit with VariableUnitaryGates'
            ' via minimization.'
        )

    @staticmethod
    def get_method_name() -> str:
        """Return the name of this method."""
        return 'minimization'

    def multi_start_instantiate_inplace(
        self,
        circuit: Circuit,
        target: UnitaryLike | StateLike | StateSystemLike,
        num_starts: int,
    ) -> None:
        """
        Instantiate `circuit` to best implement `target` with multiple starts.

        See Instantiater for more info.
        """
        target = self.check_target(target)
        start_gen = RandomStartGenerator()
        starts = start_gen.gen_starting_points(num_starts, circuit, target)
        cost_fn = self.cost_fn_gen.gen_cost(circuit, target)
        if isinstance(cost_fn, ResidualsFunction):
            cost_fn = HilbertSchmidtCostGenerator().gen_cost(circuit, target)
        params_list = [self.instantiate(circuit, target, x0) for x0 in starts]
        params = sorted(params_list, key=lambda x: cost_fn(x))[0]
        circuit.set_params(params)

    async def multi_start_instantiate_async(
        self,
        circuit: Circuit,
        target: UnitaryLike | StateLike | StateSystemLike,
        num_starts: int,
    ) -> Circuit:
        """
        Instantiate `circuit` to best implement `target` with multiple starts.

        See Instantiater for more info.
        """
        from bqskit.runtime import get_runtime
        target = self.check_target(target)
        start_gen = RandomStartGenerator()
        starts = start_gen.gen_starting_points(num_starts, circuit, target)
        cost_fn = self.cost_fn_gen.gen_cost(circuit, target)
        if isinstance(cost_fn, ResidualsFunction):
            cost_fn = HilbertSchmidtCostGenerator().gen_cost(circuit, target)
        params_list = await get_runtime().map(
            self.instantiate,
            [circuit] * num_starts,
            [target] * num_starts,
            starts,
        )
        params = sorted(params_list, key=lambda x: cost_fn(x))[0]
        circuit.set_params(params)
        return circuit
