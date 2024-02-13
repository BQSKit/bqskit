"""This module implements the Rebase2QuditGatePass."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.passes.retarget.two import Rebase2QuditGatePass
from bqskit.qis.unitary import UnitaryMatrix
from bqskit.runtime import get_runtime
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_real_number
_logger = logging.getLogger(__name__)


class AutoRebase2QuditGatePass(Rebase2QuditGatePass):
    """
    The AutoRebase2QuditGatePass class.

    Will use instantiation to change the a 2-qudit gate to a different one.
    """

    def __init__(
        self,
        max_depth: int = 3,
        max_retries: int = -1,
        success_threshold: float = 1e-8,
        cost: CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
        instantiate_options: dict[str, Any] = {},
    ) -> None:
        """
        Construct a AutoRebase2QuditGatePass.

        Args:
            max_depth (int): The maximum number of new gates to replace
                an old gate with. (Default: 3)

            max_retries (int): The number of retries for the same gate
                before we increase the maximum depth. If left as -1,
                then never increase max depth. (Default: -1)

            success_threshold (float): The distance threshold that
                determines successful termintation. Measured in cost
                described by the hilbert schmidt cost function.
                (Default: 1e-8)

            cost (CostFunction | None): The cost function that determines
                successful removal of a gate.
                (Default: HilbertSchmidtResidualsGenerator())

            instantiate_options (dict[str: Any]): Options passed directly
                to circuit.instantiate when instantiating circuit
                templates. (Default: {})

        Raises:
            ValueError: if `max_depth` is nonnegative.
        """
        if not is_integer(max_depth):
            raise TypeError(f'Expected integer, got {max_depth}.')

        if max_depth < 0:
            raise ValueError(f'Expected nonnegative depth, got: {max_depth}.')

        if not is_integer(max_retries):
            raise TypeError(f'Expected integer, got {max_retries}.')

        if not is_real_number(success_threshold):
            raise TypeError(
                'Expected real number for success_threshold'
                ', got %s' % type(success_threshold),
            )

        if not isinstance(cost, CostFunctionGenerator):
            raise TypeError(
                'Expected cost to be a CostFunctionGenerator, got %s'
                % type(cost),
            )

        if not isinstance(instantiate_options, dict):
            raise TypeError(
                'Expected dictionary for instantiate_options, got %s.'
                % type(instantiate_options),
            )
        self.max_depth = max_depth
        self.max_retries = max_retries
        self.success_threshold = success_threshold
        self.cost = cost
        self.instantiate_options: dict[str, Any] = {
            'dist_tol': self.success_threshold,
            'min_iters': 100,
        }
        self.instantiate_options.update(instantiate_options)

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        new_gates = [g for g in data.gate_set if g.num_qudits == 2]
        old_gates = [
            g for g in circuit.gate_set_no_blocks
            if g not in new_gates and g.num_qudits == 2
        ]
        circs, counts, overdrive, msgs = self.generate_new_gate_templates(
            new_gates,
            data.gate_set.get_general_sq_gate(),
        )

        instantiate_options = self.instantiate_options.copy()
        if 'seed' not in instantiate_options:
            instantiate_options['seed'] = data.seed
        _logger.debug(f'Rebasing gates from {old_gates} to {new_gates}.')

        target = self.get_target(circuit, data)

        if isinstance(target, UnitaryMatrix):
            identity = UnitaryMatrix.identity(target.dim, target.radixes)
            if target.get_distance_from(identity) < self.success_threshold:
                _logger.debug('Target is identity, returning empty circuit.')
                circuit.clear()
                return

        for g in old_gates:
            # Track retries to check for no progress
            num_retries = 0
            prev_count = circuit.count(g)

            while g in circuit.gate_set:
                # Change the seed every iteration to prevent stalls
                if instantiate_options['seed'] is not None:
                    instantiate_options['seed'] += 1

                # Check if we made progress from last loop
                gates_left = circuit.count(g)
                if prev_count == gates_left:
                    num_retries += 1
                else:
                    prev_count = gates_left
                    num_retries = 0

                # Group together a 2-qubit block composed of gates from old set
                point = self.group_near_gates(
                    circuit,
                    circuit.point(g),
                    old_gates,
                )
                circuits_with_new_gate = []
                for circ in circs:
                    circuit_copy = circuit.copy()
                    circuit_copy.replace_with_circuit(point, circ)
                    circuits_with_new_gate.append(circuit_copy)

                # If we have exceeded the number of retries, up the max depth
                if self.max_retries >= 0 and num_retries > self.max_retries:
                    _logger.debug('Exceeded max retries, increasing depth.')
                    circuit_copy = circuit.copy()
                    circuit_copy.replace_with_circuit(point, overdrive)
                    circuits_with_new_gate.append(circuit_copy)

                instantiated_circuits = await get_runtime().map(
                    Circuit.instantiate,
                    circuits_with_new_gate,
                    target=target,
                    **instantiate_options,
                )

                dists = [self.cost(c, target) for c in instantiated_circuits]

                # Find the successful circuit with the least gates
                best_index = None
                best_count = self.max_depth + 2
                for i, dist in enumerate(dists):
                    if dist < self.success_threshold:
                        if counts[i] < best_count:
                            best_index = i
                            best_count = counts[i]

                if best_index is None:
                    circuit.unfold(point)
                    continue

                _logger.debug(msgs[best_index])
                circuit.become(instantiated_circuits[best_index])
