"""This module implements the SubstitutePass."""
from __future__ import annotations

import itertools as it
import logging
from typing import Any
from typing import Callable

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gate import Gate
from bqskit.ir.operation import Operation
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.utils.typing import is_real_number
_logger = logging.getLogger(__name__)


class SubstitutePass(BasePass):
    """
    The SubstitutePass class.

    The substitute pass will attempt to use instantiation to replace gates with
    other gates.
    """

    def __init__(
        self,
        collection_filter: Callable[[Operation], bool],
        gate: Gate,
        success_threshold: float = 1e-10,
        cost: CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
        instantiate_options: dict[str, Any] = {},
    ) -> None:
        """
        Construct a SubstitutePass.

        Args:
            collection_filter (Callable[[Operation], bool]):
                A predicate that determines which operations should have
                substitution attempted on them.

            gate (Gate): The gate to try substitute in.

            success_threshold (float): The distance threshold that
                determines successful termintation. Measured in cost
                described by the hilbert schmidt cost function.
                (Default: 1e-10)

            cost (CostFunction | None): The cost function that determines
                successful removal of a gate.
                (Default: HilbertSchmidtResidualsGenerator())

            instantiate_options (dict[str: Any]): Options passed directly
                to circuit.instantiate when instantiating circuit
                templates. (Default: {})
        """
        if not callable(collection_filter):
            raise TypeError(
                'Expected callable method that maps Operations to booleans for'
                ' collection_filter, got %s.' % type(self.collection_filter),
            )

        if not isinstance(gate, Gate):
            raise TypeError(f'Expected a gate, got {type(gate)}')

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

        self.collection_filter = collection_filter
        self.gate = gate
        self.success_threshold = success_threshold
        self.cost = cost
        self.instantiate_options: dict[str, Any] = {
            'dist_tol': self.success_threshold,
        }
        self.instantiate_options.update(instantiate_options)

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""

        # Collect locations in circuit where self.init_gate exists
        points = [
            (cycle, op.location[0])
            for cycle, op in circuit.operations_with_cycles()
            if self.collection_filter(op)
        ]

        # Check substitutions are valid
        for point in points:
            if circuit[point].num_qudits < self.gate.num_qudits:
                raise RuntimeError(
                    f'Cannot substitute {circuit[point]}'
                    f' with {self.gate}.',
                )

        # Attempt substitution
        target = self.get_target(circuit, data)
        shift = 0
        for point in points:
            point = (point[0] - shift, point[1])
            _logger.info(f'Attempting substitution for operation at {point}.')
            _logger.info(f'Operation: {circuit[point]}')

            qudits = circuit[point].location
            locs = it.combinations(qudits, self.gate.num_qudits)
            og_cycle_count = circuit.num_cycles

            for loc in locs:
                _logger.debug(f'Trying location: {loc}')
                circuit_copy = circuit.copy()
                circuit_copy.replace_gate(point, self.gate, loc)
                circuit_copy = self.execute(
                    data,
                    Circuit.instantiate,
                    [circuit_copy],
                    target=target,
                    **self.instantiate_options,
                )[0]

                if self.cost(circuit_copy, target) < self.success_threshold:
                    _logger.info('Successfully substituted operation.')
                    circuit.become(circuit_copy)
                    shift += og_cycle_count - circuit.num_cycles
                    break
