"""This module implements the TreeScanningGateRemovalPass."""
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
from bqskit.runtime import get_runtime
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_real_number

_logger = logging.getLogger(__name__)


class TreeScanningGateRemovalPass(BasePass):
    """
    The TreeScanningGateRemovalPass class.

    Starting from one side of the circuit, run the following:

    Split the circuit operations into chunks of size `tree_depth`
    At every iteration:
    a. Look at the next chunk of operations
    b. Generate 2 ^ `tree_depth` circuits. Each circuit corresponds to every
    combination of whether or not to include one of the operations in the chunk.
    c. Instantiate in parallel all 2^`tree_depth` circuits
    d. Choose the circuit that has the least number of operations and move
    on to the next chunk of operations.

    This optimization is less greedy than the current
    :class:`~bqskit.passes.processing.ScanningGateRemovalPass` removal,
    which leads to much better quality circuits than ScanningGate.
    In very rare occasions, ScanningGate may be able to outperform
    TreeScan (since it is still greedy), but in general we can expect
    TreeScan to almost always outperform ScanningGate.
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
        Construct a TreeScanningGateRemovalPass.

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
                circuits will be instantiated in parallel. Note that the default
                behavior will be equivalent to normal ScanningGateRemoval
                (Default: 1)

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

        if not is_integer(tree_depth):
            raise TypeError(
                'Expected Integer type for tree_depth, got %s.'
                % type(instantiate_options),
            )

        self.tree_depth = tree_depth
        self.start_from_left = start_from_left
        self.success_threshold = success_threshold
        self.cost = cost
        self.instantiate_options: dict[str, Any] = {
            'dist_tol': self.success_threshold,
            'min_iters': 10,
            'cost_fn_gen': self.cost,
        }
        self.instantiate_options.update(instantiate_options)

    @staticmethod
    def get_tree_circs(
        orig_num_cycles: int,
        circuit_copy: Circuit,
        cycle_and_ops: list[tuple[int, Operation]],
    ) -> list[Circuit]:
        """
        Generate all circuits to be instantiated in the tree scan.

        Args:
            orig_num_cycles (int): The original number of cycles
            in the circuit. This allows us to keep track of the shift
            caused by previous deletions.

            circuit_copy (Circuit): Current state of the circuit.

            cycle_and_ops: list[(int, Operation)]: The next chunk
            of operations to be considered for deletion.

        Returns:
            list[Circuit]: A list of 2^(`tree_depth`) - 1 circuits
            that remove up to `tree_depth` operations. The circuits
            are sorted by the number of operations removed.
        """
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

        all_circs = sorted(all_circs, key=lambda x: x.num_operations)
        # Remove circuit with no gates deleted
        return all_circs[:-1]

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        instantiate_options = self.instantiate_options.copy()
        if 'seed' not in instantiate_options:
            instantiate_options['seed'] = data.seed

        start = 'left' if self.start_from_left else 'right'
        _logger.debug(f'Starting tree scanning gate removal on the {start}.')

        target = self.get_target(circuit, data)

        circuit_copy = circuit.copy()
        reverse_iter = not self.start_from_left

        ops_left = list(circuit.operations_with_cycles(reverse=reverse_iter))
        print(
            f'Starting TreeScan with tree depth {self.tree_depth}'
            f' on circuit with {len(ops_left)} gates',
        )

        while ops_left:
            chunk = ops_left[:self.tree_depth]
            ops_left = ops_left[self.tree_depth:]

            all_circs = TreeScanningGateRemovalPass.get_tree_circs(
                circuit.num_cycles, circuit_copy, chunk,
            )

            _logger.debug(
                'Attempting removal of operation of up to'
                f' {self.tree_depth} operations.',
            )

            instantiated_circuits: list[Circuit] = await get_runtime().map(
                Circuit.instantiate,
                all_circs,
                target=target,
                **instantiate_options,
            )

            dists = [self.cost(c, target) for c in instantiated_circuits]

            # Pick least count with least dist
            for i, dist in enumerate(dists):
                if dist < self.success_threshold:
                    # Log gates removed
                    gate_dict_orig = circuit_copy.gate_counts
                    gate_dict_new = instantiated_circuits[i].gate_counts
                    gates_removed = {
                        k: circuit_copy.gate_counts[k] - gate_dict_new.get(k, 0)
                        for k in gate_dict_orig.keys()
                    }
                    gates_removed = {
                        k: v for k, v in gates_removed.items() if v != 0
                    }
                    _logger.debug(
                        f'Successfully removed {gates_removed} gates',
                    )
                    circuit_copy = instantiated_circuits[i]
                    break

        circuit.become(circuit_copy)


def default_collection_filter(op: Operation) -> bool:
    return True
