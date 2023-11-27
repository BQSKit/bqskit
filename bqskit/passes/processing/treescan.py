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
import time
from bqskit.runtime import get_runtime

_logger = logging.getLogger(__name__)


class TreeScanningGateRemovalPass(BasePass):
    """
    The ScanningGateRemovalPass class.

    Starting from one side of the circuit, attempt to remove gates one-by-one.
    """

    instantiation_time = 0
    tree_creation_time = 0
    copying_time = 0
    num_instantiations = 0

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
        start = time.time()
        all_circs = [circuit_copy.copy()]
        TreeScanningGateRemovalPass.copying_time += (time.time() - start)
        for cycle, op in cycle_and_ops:
            new_circs = []
            for circ in all_circs:
                idx_shift = orig_num_cycles - circ.num_cycles
                new_cycle = cycle - idx_shift
                start = time.time()
                work_copy = circ.copy()
                TreeScanningGateRemovalPass.copying_time += (time.time() - start)
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

        while ops_left:
            chunk, ops_left = ops_left[:self.tree_depth], ops_left[self.tree_depth:]

            # Circuits of size 2 ** tree_depth - 1, 
            # ranked in order of most to fewest deletions
            start = time.time()
            all_circs = TreeScanningGateRemovalPass.get_tree_circs(circuit.num_cycles, circuit_copy, chunk)
            TreeScanningGateRemovalPass.tree_creation_time += (time.time() - start)
            all_circs = all_circs[:-1]

            _logger.debug(f'Attempting removal of operation of {self.tree_depth} operations.')
            all_circ_gate_counts = [x.num_operations for x in all_circs]
            _logger.debug(f'Circ counts: {all_circ_gate_counts}')

            start = time.time()
            instantiated_circuits = await get_runtime().map(
                    Circuit.instantiate,
                    all_circs,
                    target=target,
                    **instantiate_options,
            )
            
            TreeScanningGateRemovalPass.instantiation_time += time.time() - start
            TreeScanningGateRemovalPass.num_instantiations += len(all_circs)
            
            dists = [self.cost(c, target) for c in instantiated_circuits]
            _logger.debug(f'Distances: {dists}')

            # Pick least count with least dist
            for i, dist in enumerate(dists):
                if dist < self.success_threshold:
                    _logger.debug(f"Successfully switched to circuit {i} of {2 ** self.tree_depth}.")
                    circuit_copy = instantiated_circuits[i]
                    break

        circuit.become(circuit_copy)
        print(f"Num Instantiations: {TreeScanningGateRemovalPass.num_instantiations}")
        print(f"Instantiation Time: {TreeScanningGateRemovalPass.instantiation_time}")
        print(f"Creation Time: {TreeScanningGateRemovalPass.tree_creation_time}")
        print(f"Copying Time: {TreeScanningGateRemovalPass.copying_time}")


def default_collection_filter(op: Operation) -> bool:
    return True


#[0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 14, 16, 17, 18, 19, 20, 22, 24, 25, 26, 27, 28, 30, 32, 33, 34, 35, 36, 38, 40, 41, 42, 43, 44, 46, 48, 49, 50, 51, 52, 54, 56, 57, 58, 59, 60, 62, 64, 65, 66, 67, 68, 70, 72, 73, 74, 75, 76, 78, 80, 81, 82, 83, 84, 86, 88, 89, 90, 91, 92, 94, 96, 97, 98, 99, 100, 102, 104, 105, 106, 107, 108, 110, 112, 113, 114, 115, 116, 118, 120, 121, 122, 123, 124, 126, 128, 130, 134, 138, 139, 142, 146, 147, 150, 155, 156, 158]