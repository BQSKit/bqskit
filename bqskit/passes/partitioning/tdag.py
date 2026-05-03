"""This module defines the TDAGPartitioner pass."""
from __future__ import annotations

import copy
import logging
from typing import Callable
from typing import Sequence

from ._partitioning_utils import CachedSingleQuditIterator
from ._partitioning_utils import PriorityQueueSet
from ._partitioning_utils import SimpleCircuitPoint
from ._partitioning_utils import SingleQuditIterator
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.operation import Operation
from bqskit.ir.region import CircuitRegion
from bqskit.utils.typing import is_integer


_logger = logging.getLogger(__name__)


def default_scoring_fn(ops: list[Operation]) -> float:
    """Default parition scoring function based on QGo."""
    score = 0.0
    for op in ops:
        score += (op.num_qudits - 1) * 100 + 1
    return score


class TDAGPartitioner(BasePass):
    """
    The TDAGPartitioner Pass.

    This pass forms partitions in the circuit by scanning from left
    to right. This is based on BQSKit's ScanPartitioner.

    NOTE: This partitioner does not combine non-interacting blocks. This may
    produce several small partitions where another partitioner would produce
    one large one. This can be addressed with some simple post-processing to
    merge these blocks.

    Which blocks "interact" is determined by looking forward along each qudit,
    which also means that performance can be poor on shallow circuits, and is
    far more difficult to correct in these cases.

    References:
        Clark, J., Humble, T., & Thapliyal, H. (2023, June). Tdag:
        Tree-based directed acyclic graph partitioning for quantum circuits.
        In Proceedings of the Great Lakes Symposium on VLSI 2023 (pp. 587-592).
    """

    def __init__(
        self,
        block_size: int = 3,
        scoring_fn: Callable[[list[Operation]], float] = default_scoring_fn,
        discard_subset_groups: bool = True,
    ) -> None:
        """
        Construct a TDAGPartitioner.

        Args:
            block_size (int): Maximum size of partitioned blocks.
                (Default: 3)

            scoring_fn (Callable[[list[Operation]], float]): This function
                is used to score potential blocks. This takes in a list of
                operations in the potential block and returns a float.
                (Default: default_scoring_fn)

            discard_subset_groups (bool): Toggles evaluation of qudit groups
                which are a subset of another qudit group. Recommend disabling
                only with a custom scoring function, since it harms performance
                and doesn't improve result quality by much with the default.
                (Default: True)

        Raises:
            ValueError: If `block_size` is less than 2.
        """

        if not is_integer(block_size):
            raise TypeError(
                f'Expected integer for block_size, got {type(block_size)}.',
            )

        if block_size < 2:
            raise ValueError(
                f'Expected block_size to be greater than 2, got {block_size}.',
            )

        self.block_size = block_size
        self.scoring_fn = scoring_fn
        self.discard_subset_groups = discard_subset_groups

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""

        if self.block_size > circuit.num_qudits:
            _logger.warning(
                'Configured block size is greater than circuit size; '
                'blocking entire circuit.',
            )
            circuit.fold({
                qudit_index: (0, circuit.num_cycles - 1)
                for qudit_index in range(circuit.num_qudits)
            })
            return

        # Cache maximum cycles in circuit
        num_cycles = circuit.num_cycles

        active_qudits = set(circuit.active_qudits)

        # divider splits the circuit into partitioned and unpartitioned spaces.
        divider = [
            0 if q in active_qudits else num_cycles
            for q in range(circuit.num_qudits)
        ]

        # Stores the selected blocks as regions, initially empty
        regions: list[CircuitRegion] = []

        # potential blocks is initially empty
        potential_blocks: dict[
            frozenset[int],
            tuple[CircuitRegion, list[Operation]],
        ] = {}

        self.qudit_gate_cache: dict[int, list[int]] = {
            n: CachedSingleQuditIterator.make_gate_cache(
                circuit, 
                n, 
                multiqudit_only = True
            )
            for n in circuit.active_qudits
        }

        single_qudits = {
            qudit for qudit in range(
                circuit.num_qudits,
            ) if qudit in active_qudits and len(
                self.qudit_gate_cache[qudit],
            ) == 0
        }

        while len(single_qudits) > 0:
            region_qudits = []
            for i in range(self.block_size):
                if len(single_qudits) == 0:
                    break
                qudit = single_qudits.pop()
                divider[qudit] = num_cycles
                region_qudits.append(qudit)
            regions.append(
                CircuitRegion(
                    {qudit: (0, num_cycles - 1) for qudit in region_qudits},
                ),
            )

        # initially update all active qudits
        qudits_to_update = set(circuit.active_qudits) - single_qudits
        to_visit: PriorityQueueSet[
            tuple[
                int,
                frozenset[int],
            ]
        ] = PriorityQueueSet()

        self.gate_dependencies: dict[
            tuple[
                int,
                frozenset[int],
            ], set[int],
        ] = dict()
        self.gate_visit_set: dict[tuple[int, frozenset[int]], set[int]] = dict()

        # Form regions until there are no more gates to partition
        while any(cycle < num_cycles for cycle in divider):

            target_qudits = qudits_to_update.copy()
            qudit_groups_to_remove = list()

            # remove any blocks which include the updated qudits
            # add qudits which share a block with the updated qudits to the
            # target set (since new circuit paths may be open for these qudits)
            for group in potential_blocks.keys():
                if (not group.isdisjoint(qudits_to_update)):
                    target_qudits.update(group)
                    qudit_groups_to_remove.append(group)

            for group in qudit_groups_to_remove:
                del potential_blocks[group]

            # Update the gate dependencies using a breadth first search
            self.update_gate_dependencies(
                circuit, divider, self.gate_dependencies, self.gate_visit_set,
                target_qudits, to_visit,
            )

            # Calculate all groups of qudits starting from the target qudits
            #   that can exist based on current circuit
            new_qudit_groups = self.calculate_all_qudit_groups_recursive(
                circuit, divider, target_qudits,
            )

            # Update the map of potential blocks
            potential_blocks.update(
                self.calculate_blocks(
                    new_qudit_groups, circuit, divider,
                ),
            )

            # Select best block from current potential blocks
            best_region = self.find_best_block(potential_blocks)
            regions.append(best_region)
            _logger.info(f'Just formed region: {best_region}')

            # Remove the influence of the gates in the removed block on the
            #   gate dependencies
            to_visit = self.clear_gate_dependencies(
                circuit, divider, self.gate_dependencies, self.gate_visit_set,
                set(
                    best_region.keys(),
                ),
            )

            # Update divider
            for qudit, interval in best_region.items():
                divider[qudit] = interval[1] + 1

            # Note changed qudits
            qudits_to_update = set(best_region.keys())

        self.gate_dependencies = dict()
        self.gate_visit_set = dict()

        # Form the partitioned circuit
        circuit.become(self.fold_circuit(circuit, regions))

    def fold_circuit(
        self,
        circuit: Circuit,
        regions: list[CircuitRegion],
    ) -> Circuit:
        """Partition `circuit` into blocks described in `regions`."""
        folded_circuit = Circuit(circuit.num_qudits, circuit.radixes)

        for region in regions:
            ops_and_cycles = []
            for qudit_index, intervals in region.items():
                iterator = SingleQuditIterator(
                    SimpleCircuitPoint(intervals.lower, qudit_index),
                    circuit,
                )
                for cycle, op in iterator:
                    if cycle > intervals.upper:
                        break
                    if len(op.location) == 1 or qudit_index == min(op.location):
                        ops_and_cycles.append((cycle, op))
            ops_and_cycles.sort(key=lambda x: x[0])
            qudits = sorted(region.location)
            radixes = [circuit.radixes[q] for q in qudits]

            cgc = Circuit(len(radixes), radixes)
            for cycle, op in ops_and_cycles:
                location = [qudits.index(q) for q in op.location]
                cgc.append(
                    Operation(
                        copy.deepcopy(op.gate),
                        location,
                        op.params.copy(),
                    ),
                )

            folded_circuit.append_gate(
                CircuitGate(cgc, True),
                qudits,
                cgc.params,
            )

        return folded_circuit

    def update_gate_dependencies(
        self, circuit: Circuit, divider: list[int],
        dependencies: dict[tuple[int, frozenset[int]], set[int]],
        visit_set: dict[tuple[int, frozenset[int]], set[int]],
        target_group: set[int],
        to_visit: PriorityQueueSet[tuple[int, frozenset[int]]],
    ) -> None:
        """Sweep the `circuit` starting at the `divider` and update
        `dependencies` and `visit_set` for all qudits in the `target_group`."""

        for qudit in target_group:
            try:
                single_iter = CachedSingleQuditIterator(
                    SimpleCircuitPoint(
                        divider[qudit], qudit,
                    ), circuit, self.qudit_gate_cache[qudit],
                )
                cycle, op = next(single_iter)
            except StopIteration:
                continue

            if op.num_qudits > self.block_size:
                raise RuntimeError(
                    'TDAGPartitioner cannot handle gates larger than'
                    ' block size. You may want to use the '
                    'QuickPartitioner.',
                )

            gate = (cycle, frozenset(op.location._location))
            dependencies.setdefault(gate, set()).update(
                set(op.location._location),
            )
            visit_set.setdefault(gate, set()).add(qudit)
            to_visit.push(gate)

        while (len(to_visit) > 0):
            gate = to_visit.pop()
            (cycle, location) = gate
            # if all operation qudits are active and the gate has listed
            #   dependencies (is reachable)
            if gate in dependencies and \
                    len(dependencies[gate]) <= self.block_size and \
                    len(visit_set[gate]) == len(location):
                for qudit in location:
                    try:
                        single_iter = CachedSingleQuditIterator(
                            SimpleCircuitPoint(
                                cycle + 1, qudit,
                            ), circuit, self.qudit_gate_cache[qudit],
                        )
                        next_cycle, next_op = next(single_iter)
                    except StopIteration:
                        continue

                    if next_op.num_qudits > self.block_size:
                        raise RuntimeError(
                            'TDAGPartitioner cannot handle gates larger than'
                            ' block size. You may want to use the '
                            'QuickPartitioner.',
                        )

                    next_gate = (
                        next_cycle, frozenset(
                            next_op.location._location,
                        ),
                    )
                    dependencies.setdefault(next_gate, set()).update(
                        dependencies[gate] | set(next_op.location._location),
                    )
                    visit_set.setdefault(next_gate, set()).add(qudit)
                    to_visit.push(next_gate)

    def clear_gate_dependencies(
        self, circuit: Circuit, divider: list[int],
        dependencies: dict[tuple[int, frozenset[int]], set[int]],
        visit_set: dict[tuple[int, frozenset[int]], set[int]],
        removed_group: set[int],
    ) -> PriorityQueueSet[tuple[int, frozenset[int]]]:
        """Sweep the `circuit` starting at the `divider` and remove all qudits
        in the `removed_group` from the `dependencies` and `visit_set`."""

        to_visit: PriorityQueueSet[
            tuple[
                int,
                frozenset[int],
            ]
        ] = PriorityQueueSet()
        to_update: PriorityQueueSet[
            tuple[
                int,
                frozenset[int],
            ]
        ] = PriorityQueueSet()

        for qudit in removed_group:
            try:
                single_iter = CachedSingleQuditIterator(
                    SimpleCircuitPoint(
                        divider[qudit], qudit,
                    ), circuit, self.qudit_gate_cache[qudit],
                )
                cycle, op = next(single_iter)
            except StopIteration:
                continue
            gate = (cycle, frozenset(op.location._location))
            if (gate in dependencies):
                to_visit.push(gate)
                if (not dependencies[gate] <= removed_group):
                    dependencies[gate] -= removed_group
                    visit_set[gate] -= removed_group
                    to_update.push(gate)
                else:
                    del dependencies[gate]
                    del visit_set[gate]

        while (len(to_visit) > 0):
            gate = to_visit.pop()
            (cycle, location) = gate
            for qudit in location:
                try:
                    single_iter = CachedSingleQuditIterator(
                        SimpleCircuitPoint(
                            cycle + 1, qudit,
                        ), circuit, self.qudit_gate_cache[qudit],
                    )
                    next_cycle, next_op = next(single_iter)
                except StopIteration:
                    continue
                next_gate = (next_cycle, frozenset(next_op.location._location))
                if (next_gate in dependencies):
                    to_visit.push(next_gate)
                    if (not dependencies[next_gate] <= removed_group):
                        dependencies[next_gate] -= removed_group
                        visit_set[next_gate] -= removed_group
                        to_update.push(next_gate)
                    else:
                        del dependencies[next_gate]
                        del visit_set[next_gate]

        return to_update

    def gate_dependencies_valid(self, gate: tuple[int, frozenset[int]]) -> bool:
        (cycle, location) = gate
        return gate in self.gate_dependencies and \
            len(self.gate_visit_set[gate]) == len(location)

    def calculate_all_qudit_groups_recursive(
        self,
        circuit: Circuit,
        divider: list[int],
        target_qudits: set[int],
    ) -> list[frozenset[int]]:
        """Calculates all relevant groups of qudits using the method outlined in
        the paper."""
        node_results: list[set[frozenset[int]]] = [
            set() for _ in range(self.block_size + 1)
        ]
        to_remove: set[frozenset[int]] = set()

        for n in target_qudits:
            node_next: list[set[frozenset[int]]] = [
                set()
                for _ in range(self.block_size)
            ]
            node_next[-1].add(frozenset({n}))

            visited: set[tuple[int, frozenset[int]]] = set()
            starting_point = SimpleCircuitPoint(divider[n], n)
            node_current = self.calculate_all_qudit_groups_recursive_inner(
                circuit, divider, target_qudits, node_next,
                starting_point, visited, to_remove,
            )

            for i in range(len(node_current)):
                node_results[i] |= node_current[i]

        # remove the null set from the results
        del node_results[-1]

        # flatten results into one list and
        # remove all superceded groups
        qudit_groups: list[frozenset[int]] = [
            qudit_group for group_set in node_results
            for qudit_group in group_set if qudit_group not in to_remove
        ]

        return qudit_groups

    def calculate_all_qudit_groups_recursive_inner(
        self,
        circuit: Circuit,
        divider: list[int],
        target_qudits: set[int],
        node_current: list[set[frozenset[int]]],
        starting_point: SimpleCircuitPoint,
        visited: set[tuple[int, frozenset[int]]],
        to_remove: set[frozenset[int]],
    ) -> list[set[frozenset[int]]]:
        node_results: list[set[frozenset[int]]] = [
            set() for _ in range(len(node_current) + 1)
        ]

        for i in range(len(node_current)):
            node_results[i] |= node_current[i]

        iterator = CachedSingleQuditIterator(
            starting_point,
            circuit,
            self.qudit_gate_cache[starting_point.qudit],
        )
        for cycle, op in iterator:
            if (len(node_current) <= 1):
                break

            gate = (cycle, frozenset(op.location._location))
            if not self.gate_dependencies_valid(gate):
                break

            other_qudit = op.location._location[0] if op.location._location[
                0
            ] != starting_point.qudit else op.location._location[1]

            node_next: list[set[frozenset[int]]] = [
                set()
                for _ in range(len(node_current) - 1)
            ]
            any_paths = False

            for set_index in range(len(node_next)):
                for qubit_set in node_current[set_index + 1]:
                    combined_set = qubit_set | self.gate_dependencies[gate]
                    # skip sets which do not get bigger, or are the same as
                    # the dependencies and have been visited to avoid an
                    # explosion in runtime when gates repeat (i.e.
                    #   consecutive gates between 0 and 1)
                    skip = len(combined_set) == len(qubit_set) or \
                        len(combined_set) == \
                        len(self.gate_dependencies[gate]) and \
                        gate in visited
                    if (len(combined_set) <= self.block_size) and not skip:
                        any_paths = True
                        node_next[
                            self.block_size - len(combined_set)
                        ].add(combined_set)
                        if self.discard_subset_groups:
                            to_remove.add(qubit_set)

            visited.add(gate)
            if any_paths:
                next_starting_point = SimpleCircuitPoint(
                    cycle + 1, other_qudit,
                )
                node_current = self.calculate_all_qudit_groups_recursive_inner(
                    circuit, divider, target_qudits, node_next,
                    next_starting_point, visited, to_remove,
                )
                for i in range(len(node_current)):
                    node_results[i] |= node_current[i]

        return node_results

    def calculate_blocks(
        self,
        qudit_groups: Sequence[frozenset[int]],
        circuit: Circuit,
        divider: Sequence[int],
    ) -> dict[frozenset[int], tuple[CircuitRegion, list[Operation]]]:
        """Calculate the initial blocks for all qudit groups."""

        blocks: dict[
            frozenset[int],
            tuple[CircuitRegion, list[Operation]],
        ] = {}

        for qg in qudit_groups:
            blocks[qg] = self.calculate_block(
                qg, circuit, [divider[q] for q in qg],
            )

        return blocks

    def find_best_block(
        self,
        potential_blocks: dict[
            frozenset[int],
            tuple[CircuitRegion, list[Operation]],
        ],
    ) -> CircuitRegion:
        """Find and return the current best scoring block."""
        return sorted(
            list(potential_blocks.values()),
            key=lambda x: self.scoring_fn(x[1]),
        )[-1][0]

    def calculate_block(
        self,
        qudit_group: frozenset[int],
        circuit: Circuit,
        starting_cycles: Sequence[int],
    ) -> tuple[CircuitRegion, list[Operation]]:
        """
        Calculate the best block for `qudit_group` right of the divider.

        Args:
            qudit_group (Sequence[int]): The block's qudits.

            circuit (Circuit): The circuit to form a block in.

            starting_cycles (Sequence[int]): Where to start scanning
                right from.

        Returns:
            (tuple[CircuitRegion, list[Operation]]): The formed region
                and the operations in the block.
        """
        in_qudits = {q for q in qudit_group}
        stopped_cycles = {q: circuit.num_cycles for q in qudit_group}
        op_list: list[Operation] = []
        to_visit: PriorityQueueSet[
            tuple[
                int,
                frozenset[int],
            ]
        ] = PriorityQueueSet()

        # get the first gate along each qudit and schedule it for visitation
        # if there isn't another gate, remove the qudit from the active qudits
        for qudit, starting_cycle in zip(qudit_group, starting_cycles):
            try:
                single_iter = CachedSingleQuditIterator(
                    SimpleCircuitPoint(
                        starting_cycle, qudit,
                    ), circuit, self.qudit_gate_cache[qudit],
                )
                cycle, op = next(single_iter)
            except StopIteration:
                in_qudits.remove(qudit)
                continue
            gate = (cycle, frozenset(op.location._location))
            to_visit.push(gate)

        # visit each gate in topological order (using the PriorityQueueSet),
        # creating wedge-shaped blocks by removing qudits from the active set
        # when a gate which cannot be included is encountered; continue until
        # there are no active qudits or no gates left to visit
        while (len(in_qudits) > 0 and len(to_visit) > 0):
            gate = to_visit.pop()
            (cycle, location) = gate
            # if all operation qudits are active and the gate has valid
            #   dependencies (is reachable)
            if (location <= in_qudits and self.gate_dependencies_valid(gate)):
                # put current gate in op_list
                op_list.append(
                    circuit._circuit[cycle][
                        next(
                            iter(location),
                        )
                    ],  # type: ignore
                )
                # schedule the next gate along each qudit for visitation
                for qudit in location:
                    try:
                        single_iter = CachedSingleQuditIterator(
                            SimpleCircuitPoint(
                                cycle + 1, qudit,
                            ), circuit, self.qudit_gate_cache[qudit],
                        )
                        next_cycle, next_op = next(single_iter)
                    except StopIteration:
                        in_qudits.remove(qudit)
                        continue
                    next_gate = (
                        next_cycle, frozenset(
                            next_op.location._location,
                        ),
                    )
                    to_visit.push(next_gate)
            else:
                for qudit in location:
                    if qudit in in_qudits:
                        stopped_cycles[qudit] = cycle
                in_qudits -= location

        return (
            CircuitRegion({
                q: (starting_cycles[i], stopped_cycles[q] - 1)
                for i, q in enumerate(qudit_group)
                if stopped_cycles[q] - 1 >= starting_cycles[i]
            }),
            op_list,
        )
