"""This module implements the ScanningGateRemovalPass."""
from __future__ import annotations

import logging
from typing import Any
from typing import Callable

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.operation import Operation
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.utils.typing import is_real_number
_logger = logging.getLogger(__name__)


class ScanningGateRemovalPass(BasePass):
    """
    The ScanningGateRemovalPass class.

    Starting from one side of the circuit, attempt to remove gates one-by-one.
    """

    def __init__(
        self,
        start_from_left: bool = True,
        success_threshold: float = 1e-10,
        cost: CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
        instantiate_options: dict[str, Any] = {},
        collection_filter: Callable[[Operation], bool] | None = None,
    ) -> None:
        """
        Construct a ScanningGateRemovalPass.

        Args:
            start_from_left (bool): Determines where the scan starts
                attempting to remove gates from. If True, scan goes left
                to right, otherwise right to left. (Default: True)

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

            collection_filter (Callable[[Operation], bool] | None):
                A predicate that determines which operations should be
                attempted to be removed. Called with each operation
                in the circuit. If this returns true, this pass will
                attempt to remove that operation. Defaults to all
                operations.
        """

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

        self.collection_filter = collection_filter or default_collection_filter

        if not callable(self.collection_filter):
            raise TypeError(
                'Expected callable method that maps Operations to booleans for'
                ' collection_filter, got %s.' % type(self.collection_filter),
            )

        self.start_from_left = start_from_left
        self.success_threshold = success_threshold
        self.cost = cost
        self.instantiate_options: dict[str, Any] = {
            'dist_tol': self.success_threshold,
            'min_iters': 100,
        }
        self.instantiate_options.update(instantiate_options)

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        start = 'left' if self.start_from_left else 'right'
        _logger.debug(f'Starting scan gate removal from {start}.')

        target = circuit.get_unitary()

        circuit_copy = circuit.copy()
        reverse_iter = not self.start_from_left
        for cycle, op in circuit.operations_with_cycles(reverse=reverse_iter):

            if not self.collection_filter(op):
                _logger.debug(f'Skipping operation {op} at cycle {cycle}.')
                continue

            _logger.info(f'Attempting removal of operation at cycle {cycle}.')
            _logger.debug(f'Operation: {op}')

            working_copy = circuit_copy.copy()

            # If removing gates from the left, we need to track index changes.
            if self.start_from_left:
                idx_shift = circuit.num_cycles
                idx_shift -= working_copy.num_cycles
                cycle -= idx_shift

            working_copy.pop((cycle, op.location[0]))
            working_copy.instantiate(target, **self.instantiate_options)

            if self.cost(working_copy, target) < self.success_threshold:
                _logger.info('Successfully removed operation.')
                circuit_copy = working_copy

        circuit.become(circuit_copy)


def default_collection_filter(op: Operation) -> bool:
    return True
