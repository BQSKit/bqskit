"""This module defines the GTQCPartitioner pass."""
from __future__ import annotations

import copy
import itertools
import logging
from typing import Callable
from typing import Iterable
from typing import Sequence

from ._partitioning_utils import CachedSingleQuditIterator
from ._partitioning_utils import PriorityQueueSet
from ._partitioning_utils import SingleQuditIterator
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.operation import Operation
from bqskit.ir.region import CircuitRegion
from bqskit.utils.typing import is_integer

_logger = logging.getLogger(__name__)


def default_scoring_fn(ops: Iterable[tuple[int, Operation]]) -> float:
    """Default parition scoring function based on QGo."""
    score = 0.0
    for cycle, op in ops:
        score += (op.num_qudits - 1) * 100 + 1
    return score


class GTQCPartitioner(BasePass):
    """
    The GTQCPartitioner Pass.

    This pass forms partitions in the circuit by scanning from left
    to right. This is based on BQSKit's ScanPartitioner.

    NOTE: This partitioner does not combine non-interacting blocks. This may
    produce several small partitions where another partitioner would produce
    one large one. This can be addressed with some simple post-processing to
    merge these blocks.

    References:
        Clark, J., Humble, T. S., & Thapliyal, H. (2023, September). Gtqcp:
        Greedy topology-aware quantum circuit partitioning. In 2023 IEEE
        International Conference on Quantum Computing and Engineering (QCE)
        (Vol. 1, pp. 739-744). IEEE.
    """

    def __init__(
        self,
        block_size: int = 3,
        scoring_fn: Callable[
            [Iterable[tuple[int, Operation]]],
            float,
        ] = default_scoring_fn,
        discard_subset_groups: bool = True,
    ) -> None:
        """
        Construct a GTQCPartitioner.

        Args:
            block_size (int): Maximum size of partitioned blocks.
                (Default: 3)

            scoring_fn (Callable[[list[tuple[int, Operation]]], float]): This
                function is used to score potential blocks. This takes in a
                list of tuples of cycle and operation in the potential block
                and returns a float.
                (Default: default_scoring_fn)

            discard_subset_groups (bool): Toggles evaluation of qudit groups
                which are a subset of another qudit group. Recommend disabling
                only with a custom scoring function, since it harms performance
                and doesn't improve result quality with the default.
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
            tuple[dict[int, tuple[int, int]], Callable[[], float]],
        ] = {}

        # the gate cache contains all multiqudit gates for each qudit,
        #   in cycle order
        self.qudit_gate_cache: dict[int, list[int]] = {
            n: CachedSingleQuditIterator.make_gate_cache(
                circuit,
                n,
                multiqudit_only=True,
            )
            for n in circuit.active_qudits
        }
        # The op cache contains all gates/operations along a qudit for
        #   which the qudit is the "primary" in cycle order.
        # This cache is used to enumerate all operations within a region
        #   without using a set. This is accomplished by designating a
        #   "primary" qudit for each gate, and only listing gates under
        #   their primary qudit
        # We arbitrarily designate the qudit with the min index as the "primary"
        self.qudit_op_cache: dict[int, list[int]] = {
            n: [
                cycle for cycle,
                op in SingleQuditIterator(circuit, n, 0)
                if n == min(op.location)
            ]
            for n in circuit.active_qudits
        }

        single_qudits = {
            qudit for qudit in range(
                circuit.num_qudits,
            ) if qudit in active_qudits and len(
                self.qudit_gate_cache[qudit],
            ) == 0
        }

        # block any qudits which do not have multiqudit gates,
        #   as densely as possible
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
            #   target set (since new circuit paths may be open for
            #   these qudits)
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
            new_qudit_groups = self.calculate_all_bounds_recursive(
                circuit, divider, target_qudits,
            )

            # Update the map of potential blocks
            potential_blocks.update(
                self.calculate_blocks_from_boundaries(
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
                circuit, divider, self.gate_dependencies,
                self.gate_visit_set, set(
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
        circuit.become(self.fold_circuit(circuit, regions), False)

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
                iterator = CachedSingleQuditIterator(
                    circuit,
                    self.qudit_op_cache[qudit_index],
                    qudit_index,
                    intervals.lower,
                    intervals.upper + 1,
                )
                for cycle, op in iterator:
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
                    circuit,
                    self.qudit_gate_cache[qudit],
                    qudit,
                    divider[qudit],
                )
                cycle, op = next(single_iter)
            except StopIteration:
                continue

            if op.num_qudits > self.block_size:
                raise RuntimeError(
                    'GTQCPartitioner cannot handle gates larger than'
                    ' block size. You may want to use the '
                    'QuickPartitioner.',
                )

            gate = (cycle, frozenset(op.location))
            dependencies.setdefault(gate, set()).update(set(op.location))
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
                            circuit,
                            self.qudit_gate_cache[qudit],
                            qudit,
                            cycle + 1,
                        )
                        next_cycle, next_op = next(single_iter)
                    except StopIteration:
                        continue

                    if next_op.num_qudits > self.block_size:
                        raise RuntimeError(
                            'GTQCPartitioner cannot handle gates larger than'
                            ' block size. You may want to use the '
                            'QuickPartitioner.',
                        )

                    next_gate = (next_cycle, frozenset(next_op.location))
                    dependencies.setdefault(next_gate, set()).update(
                        dependencies[gate] | set(next_op.location),
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
                    circuit,
                    self.qudit_gate_cache[qudit],
                    qudit,
                    divider[qudit],
                )
                cycle, op = next(single_iter)
            except StopIteration:
                continue
            gate = (cycle, frozenset(op.location))
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
                        circuit,
                        self.qudit_gate_cache[qudit],
                        qudit,
                        cycle + 1,
                    )
                    next_cycle, next_op = next(single_iter)
                except StopIteration:
                    continue
                next_gate = (next_cycle, frozenset(next_op.location))
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

    def calculate_all_bounds_recursive(
        self,
        circuit: Circuit,
        divider: list[int],
        target_qudits: set[int],
    ) -> dict[frozenset[int], dict[int, int]]:
        """Finds a set of relevant qudit groups and partial partition boundary
        for the given `circuit`, `divider`, and `target_qudits`."""
        blocks: dict[frozenset[int], dict[int, int]] = dict()

        target_set = frozenset(target_qudits)

        in_qudits: frozenset[int] = frozenset()
        bounds: dict[int, int] = dict()
        to_remove: set[frozenset[int]] = set()

        self.calculate_all_bounds_recursive_inner(
            circuit, divider, blocks, to_remove, target_set,
            in_qudits, bounds,
        )

        # remove all superceded groups
        for group in to_remove:
            blocks.pop(group, None)

        # remove the empty set if it is present
        if frozenset() in blocks:
            blocks.pop(frozenset())

        return blocks

    def calculate_all_bounds_recursive_inner(
        self,
        circuit: Circuit,
        divider: list[int],
        blocks: dict[frozenset[int], dict[int, int]],
        to_remove: set[frozenset[int]],
        target_qudits: frozenset[int],
        in_qudits: frozenset[int],
        bounds: dict[int, int],
    ) -> None:
        for qudit in target_qudits:
            # we start with the existing dependencies
            best_dependencies: frozenset[int] | set[int] = in_qudits
            # create an iterator along the target qudit from the front of
            #   the divider
            qudit_iter = CachedSingleQuditIterator(
                circuit,
                self.qudit_gate_cache[qudit],
                qudit,
                divider[qudit],
            )
            # Find the furthest gate which leaves the
            #   dependencies at <= k qudits
            while (True):
                # this try block stops the loop at the end of the circuit
                try:
                    (cycle, op) = next(qudit_iter)
                except StopIteration:
                    cycle = circuit.num_cycles
                    break

                # this if statement excludes gates which do not have a complete
                #   dependency set (they cannot be reached with the current
                #   k limit)
                gate = (cycle, frozenset(op.location))
                if not self.gate_dependencies_valid(gate):
                    break
                dependencies = self.gate_dependencies[gate] | best_dependencies
                if len(dependencies) <= self.block_size:
                    best_dependencies = dependencies
                else:
                    break
            best_dependencies = frozenset(best_dependencies)
            recur = best_dependencies not in blocks
            # keep track of the qudit bounds, so that the information is
            #   available for forming partitions later
            blocks.setdefault(best_dependencies, dict()).update(bounds)
            if qudit not in blocks[best_dependencies]:
                blocks[best_dependencies][qudit] = cycle - 1
            new_targets: frozenset[int] = best_dependencies - \
                (in_qudits | {qudit})
            # schedule the old group for removal, since it is a subset of this
            # group (and therefore the resulting partition is no larger)
            if self.discard_subset_groups and len(new_targets) > 0:
                to_remove.add(frozenset(in_qudits))
            # if we have not seen this group before, there are new qudits, and
            # there is room for more qudits, make a recursive call
            if recur:
                if len(best_dependencies) < self.block_size and \
                        len(new_targets) > 0:
                    new_bounds = bounds.copy()
                    if qudit not in new_bounds:
                        new_bounds[qudit] = cycle - 1
                    self.calculate_all_bounds_recursive_inner(
                        circuit, divider, blocks, to_remove, new_targets,
                        best_dependencies, new_bounds,
                    )

    def calculate_blocks_from_boundaries(
        self,
        qudit_groups: dict[frozenset[int], dict[int, int]],
        circuit: Circuit,
        divider: Sequence[int],
    ) -> dict[
        frozenset[int], tuple[
            dict[int, tuple[int, int]], Callable[[], float],
        ],
    ]:
        """Calculates the set of possible partitions given a `circuit`,
        `divider` and set of qudit groups with partial boundary
        `qudit_groups`."""

        # stores the calculated blocks for each group, which are composed of:
        #   - a cycle range along each qudit representing the circuit region
        #   - a set of operations contained by the region
        #   - a list containing a single int which is the score for the block
        blocks: dict[
            frozenset[int],
            tuple[
                dict[int, tuple[int, int]], Callable[[], float],
            ],
        ] = {}

        # iterate over all qudit groups and calculate the boundary for any
        #   qudits without a known boundary
        for group, ends in qudit_groups.items():
            missing = group - frozenset(ends.keys())
            # iterate over qudits with no boundary
            for qudit in missing:
                qudit_iter = CachedSingleQuditIterator(
                    circuit,
                    self.qudit_gate_cache[qudit],
                    qudit,
                    divider[qudit],
                )
                cycle = circuit.num_cycles
                while (True):
                    try:
                        (cycle, op) = next(qudit_iter)
                    except StopIteration:
                        cycle = circuit.num_cycles
                        break
                    gate = (cycle, frozenset(op.location))
                    if not self.gate_dependencies_valid(gate):
                        break
                    if not self.gate_dependencies[gate] <= group:
                        break
                ends[qudit] = cycle - 1

            op_list = [
                CachedSingleQuditIterator(
                    circuit,
                    self.qudit_op_cache[qudit],
                    qudit,
                    divider[qudit],
                    ends[qudit] + 1,
                )
                for qudit in group
            ]

            blocks[group] = (
                {
                    q: (divider[q], ends[q])
                    for q in group
                },
                lambda op_list=op_list:     # type: ignore[misc]
                self.scoring_fn(itertools.chain.from_iterable(op_list)),
            )

        return blocks

    def find_best_block(
        self,
        potential_blocks: dict[
            frozenset[int],
            tuple[
                dict[int, tuple[int, int]], Callable[[], float],
            ],
        ],
    ) -> CircuitRegion:
        """Find and return the current best scoring block."""
        region = sorted(
            list(potential_blocks.values()),
            key=lambda x: x[1](),
        )[-1][0]
        return CircuitRegion(region)
