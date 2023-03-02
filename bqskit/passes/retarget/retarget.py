from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from bqskit.compiler.basepass import BasePass

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit
    from bqskit.compiler.passdata import PassData


_logger = logging.getLogger(__name__)


class RetargetGatesPass(BassPass):
    """Retarget the gates in the circuit to match the pass model."""

    def __init__(
        self,
        synthesis_epsilon: float = 1e-10,
        max_synthesis_size: int = 4,
        error_threshold: float | None = None,
        error_sim_size: int = 8,
    ) -> None:
        pass

    async def run(self, circuit: Circuit, data: PassData) -> None:
        pass
        # circuit is single-qudit
        # qudit mismatch


class RetargetManyQuditGatesPass(BasePass):
    """Retargets many-qudit gates in the circuit to match the pass model."""


class RetargetTwoQuditGatesPass(BasePass):
    """Retargets two-qudit gates in the circuit to match the pass model."""


class RetargetSingleQuditGatesPass(BasePass):
    """Retargets single-qudit gates in the circuit to match the pass model."""

    async def run(self, circuit: Circuit, data: PassData) -> None:
        target_gates = {g for g in data.model.gate_set if g.num_qudits == 1}
        input_gates = {g for g in circuit.gate_set if g.num_qudits == 1}

        if len(input_gates) == 0 or input_gates.issubset(target_gates):
            _logger.info('No single-qudit gates to retarget')
            return

        if len(target_gates) == 0:
            _logger.warning('No valid target single-qudit gate in model.')

        _logger.info(
            'Retargeting single-qudit gates from {input_gates} to {target_gates}.',
        )

    if len(sq_gates) == 0:
        return NOOPPass()

    if all(g.is_constant() for g in sq_gates):
        _logger.warn(
            'The standard workflow with BQSKit may have trouble'
            ' targeting gate sets containing no parameterized'
            ' single-qudit gates.',
        )
        # TODO: Implement better retargeting techniques for constant gate sets

    instantiate_options = {
        'method': 'minimization',
        'minimizer': ScipyMinimizer(),
        'cost_fn_gen': HilbertSchmidtCostGenerator(),
    }
    layer_generator = SingleQuditLayerGenerator(sq_gates, True)
    core_sq_rebase: BasePass = NOOPPass()
    if len(sq_gates) == 1:
        if sq_gates[0] == U3Gate():
            core_sq_rebase = U3Decomposition()

    elif len(sq_gates) >= 2:
        if RZGate() in sq_gates and SqrtXGate() in sq_gates:
            core_sq_rebase = ZXZXZDecomposition()

    if isinstance(core_sq_rebase, NOOPPass):
        core_sq_rebase = QSearchSynthesisPass(
            layer_generator=layer_generator,
            heuristic_function=DijkstraHeuristic(),
            instantiate_options=instantiate_options,
        )

    return PassGroup([
        IfThenElsePass(
            NotPredicate(SinglePhysicalPredicate()),
            [
                GroupSingleQuditGatePass(),
                ForEachBlockPass([
                    IfThenElsePass(
                        NotPredicate(SinglePhysicalPredicate()),
                        core_sq_rebase,
                    ),
                ]),
                UnfoldPass(),
            ],
        ),
    ])
