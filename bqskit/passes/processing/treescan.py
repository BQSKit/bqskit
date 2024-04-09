"""This module implements the ScanningGateRemovalPass."""
from __future__ import annotations

import logging
from typing import Any
from typing import Callable

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.operation import Operation
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.utils.typing import is_real_number
from bqskit.runtime import get_runtime

_logger = logging.getLogger(__name__)


class TreeScanningGateRemovalPass(BasePass):
    """
    The ScanningGateRemovalPass class.

    Starting from one side of the circuit, attempt to remove gates one-by-one.
    """

    def __init__(
        self,
        start_from_left: bool = True,
        success_threshold: float = 1e-8,
        cost: CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
        instantiate_options: dict[str, Any] = {},
        tree_depth: int = 1,
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
                (Default: 1e-8)

            cost (CostFunction | None): The cost function that determines
                successful removal of a gate.
                (Default: HilbertSchmidtResidualsGenerator())

            instantiate_options (dict[str: Any]): Options passed directly
                to circuit.instantiate when instantiating circuit
                templates. (Default: {})

            tree_depth (int): The depth of the tree of potential 
                solutions to instantiate. Note that 2^(tree_depth) - 1 
                circuits will be instantiated in parallel. 

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

        self.tree_depth = tree_depth
        self.start_from_left = start_from_left
        self.success_threshold = success_threshold
        self.cost = cost
        self.instantiate_options: dict[str, Any] = {
            'dist_tol': self.success_threshold,
            'min_iters': 100,
            'cost_fn_gen': self.cost,
        }
        self.instantiate_options.update(instantiate_options)

    # Implement recursively for now, if slow then fix
    def get_tree_circs(orig_num_cycles, circuit_copy: Circuit, cycle_and_ops: list[tuple[int, Operation]]) -> list[Circuit]:
        all_circs = [circuit_copy.copy()]
        for cycle, op in cycle_and_ops:
            new_circs = []
            for circ in all_circs:
                idx_shift = orig_num_cycles - circ.num_cycles
                new_cycle = cycle - idx_shift
                work_copy = circ.copy()
                work_copy.pop((new_cycle, op.location[0]))
                new_circs.append(work_copy)
                new_circs.append(circ)

            all_circs = new_circs

        all_circs = sorted(all_circs, key= lambda x: x.num_operations)

        return all_circs



    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        instantiate_options = self.instantiate_options.copy()
        if 'seed' not in instantiate_options:
            instantiate_options['seed'] = data.seed

        start = 'left' if self.start_from_left else 'right'
        _logger.debug(f'Starting scanning gate removal on the {start}.')

        target = self.get_target(circuit, data)
        # target = None

        circuit_copy = circuit.copy()
        reverse_iter = not self.start_from_left

        ops_left = list(circuit.operations_with_cycles(reverse=reverse_iter))
        print(f"Starting Scan with tree depth {self.tree_depth} on circuit with {len(ops_left)} gates")

        while ops_left:
            chunk, ops_left = ops_left[:self.tree_depth], ops_left[self.tree_depth:]

            # Circuits of size 2 ** tree_depth - 1, 
            # ranked in order of most to fewest deletions
            all_circs = TreeScanningGateRemovalPass.get_tree_circs(circuit.num_cycles, circuit_copy, chunk)
            # Remove circuit with no gates deleted
            all_circs = all_circs[:-1]

            _logger.debug(f'Attempting removal of operation of {self.tree_depth} operations.')

            instantiated_circuits = await get_runtime().map(
                    Circuit.instantiate,
                    all_circs,
                    target=target,
                    **instantiate_options,
            )
            
            dists = [self.cost(c, target) for c in instantiated_circuits]
            _logger.debug(f'Circuit distances: {dists}')

            # Pick least count with least dist
            for i, dist in enumerate(dists):
                if dist < self.success_threshold:
                    _logger.debug(f"Successfully switched to circuit {i} of {2 ** self.tree_depth}.")
                    circuit_copy = instantiated_circuits[i]
                    break

        circuit.become(circuit_copy)


def default_collection_filter(op: Operation) -> bool:
    return True