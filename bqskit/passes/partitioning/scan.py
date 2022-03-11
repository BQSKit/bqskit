"""This module defines the ScanPartitioner pass."""
from __future__ import annotations

import logging
from typing import Any
from typing import Callable
from typing import Iterator
from typing import Sequence
from typing import Tuple

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.machine import MachineModel
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


class ScanPartitioner(BasePass):
    """
    The ScanPartitioner Pass.

    This pass forms partitions in the circuit by scanning from left
    to right. This is based on the partitioner from QGo.

    References:
        Wu, Xin-Chuan, et al. "QGo: Scalable Quantum Circuit Optimization
        Using Automated Synthesis." arXiv preprint arXiv:2012.09835 (2020).
    """

    def __init__(
        self,
        block_size: int,
        scoring_fn: Callable[[list[Operation]], float] = default_scoring_fn,
    ) -> None:
        """
        Construct a ScanPartitioner.

        Args:
            block_size (int): Maximum size of partitioned blocks.
                (Default: 3)

            scoring_fn (Callable[[list[Operation]], float]): This function
                is used to score potential blocks. This takes in a list of
                operations in the potential block and returns a float.
                (Default: default_scoring_fn)

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

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""

        if self.block_size > circuit.num_qudits:
            _logger.warning(
                'Configured block size is greater than circuit size; '
                'blocking entire circuit.',
            )
            circuit.fold({
                qudit_index: (0, circuit.num_cycles)
                for qudit_index in range(circuit.num_qudits)
            })
            return

        # Cache maximum cycles in circuit
        num_cycles = circuit.num_cycles

        # All groups of qudits that can exist in viable partitions
        qudit_groups = self.calculate_qudit_groups(circuit)

        # Map from qudit indices to a list of groups they exist in
        qudit_group_map = self.calculate_qudit_group_map(qudit_groups)

        # Map from qudit groups to current best block for that group
        potential_blocks = self.calculate_initial_blocks(qudit_groups, circuit)

        # divider splits the circuit into partitioned and unpartitioned spaces.
        divider = [
            0 if q in qudit_group_map.keys() else num_cycles
            for q in range(circuit.num_qudits)
        ]

        # Stores the selected blocks as regions
        regions: list[CircuitRegion] = []

        # Form regions until there are no more gates to partition
        while any(cycle < num_cycles for cycle in divider):

            # Select best block from current potential blocks
            best_region = self.find_best_block(potential_blocks)
            regions.append(best_region)
            _logger.info(f'Just formed region: {best_region}')

            # Update divider
            for qudit, interval in best_region.items():
                divider[qudit] = interval[1] + 1

            # Update adjacent groups
            adjacent_groups = set()
            for qudit in best_region.keys():
                adjacent_groups.update(qudit_group_map[qudit])

            for qudit_group in adjacent_groups:
                starting_cycles = [divider[q] for q in qudit_group]
                new_block = self.calculate_block(
                    qudit_group, circuit, starting_cycles,
                )
                potential_blocks[qudit_group] = new_block

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
            ops_and_cycles = list({
                (point[0], circuit[point])
                for point in region.points
                if not circuit.is_point_idle(point)
            })
            ops_and_cycles.sort(key=lambda x: (x[0], *x[1].location))
            ops = [op for _, op in ops_and_cycles]
            qudits = list(set(sum((tuple(op.location) for op in ops), ())))
            qudits = sorted(qudits)
            radixes = [circuit.radixes[q] for q in qudits]

            cgc = Circuit(len(radixes), radixes)
            for op in ops:
                location = [qudits.index(q) for q in op.location]
                cgc.append(Operation(op.gate, location, op.params))

            folded_circuit.append_gate(
                CircuitGate(cgc, True),
                qudits,
                cgc.params,
            )

        return folded_circuit

    def calculate_qudit_groups(self, circuit: Circuit) -> list[tuple[int, ...]]:
        """Calculate the groups of qudits that grouped in a partition."""
        # Get initial set from circuit connectivity
        model = MachineModel(circuit.num_qudits, circuit.coupling_graph)
        qudit_groups: list[tuple[int, ...]] = []
        for i in reversed(range(self.block_size)):
            potential_groups = [tuple(x) for x in model.get_locations(i + 1)]
            for potential_group in potential_groups:
                should_add_group = True
                for qudit_group in qudit_groups:
                    if all(x in qudit_group for x in potential_group):
                        should_add_group = False
                        break
                if should_add_group:
                    qudit_groups.append(potential_group)

        # Prune groups
        active_qudits = circuit.active_qudits
        qudits_in_groups: set[int] = set()
        qudit_groups_to_remove = []
        qudit_groups_to_append = []
        for qudit_group in qudit_groups:
            if all(q in active_qudits for q in qudit_group):
                qudits_in_groups.update(qudit_group)
            else:
                qudit_groups_to_remove.append(qudit_group)
                group = tuple(q for q in qudit_group if q in active_qudits)
                if len(group) > 0:
                    qudit_groups_to_append.append(group)
                    qudits_in_groups.update(group)

        for qudit_group in qudit_groups_to_remove:
            qudit_groups.remove(qudit_group)

        for qudit_group in qudit_groups_to_append:
            qudit_groups.append(qudit_group)

        # Add groups for individual active qudits
        for q in active_qudits:
            if q not in qudits_in_groups:
                qudit_groups.append((q,))

        return qudit_groups

    def calculate_qudit_group_map(
        self,
        qudit_groups: list[tuple[int, ...]],
    ) -> dict[int, list[tuple[int, ...]]]:
        """Calculate a map from qudits to qudit_groups containing that qudit."""
        qudit_to_group_map: dict[int, list[tuple[int, ...]]] = {}
        for qudit_group in qudit_groups:
            for qudit in qudit_group:
                if qudit not in qudit_to_group_map:
                    qudit_to_group_map[qudit] = []
                qudit_to_group_map[qudit].append(qudit_group)
        return qudit_to_group_map

    def calculate_initial_blocks(
        self,
        qudit_groups: Sequence[tuple[int, ...]],
        circuit: Circuit,
    ) -> dict[tuple[int, ...], tuple[CircuitRegion, list[Operation]]]:
        """Calculate the initial blocks for all qudit groups."""

        blocks: dict[
            tuple[int, ...],
            tuple[CircuitRegion, list[Operation]],
        ] = {}

        for qg in qudit_groups:
            blocks[qg] = self.calculate_block(qg, circuit, [0] * len(qg))

        return blocks

    def find_best_block(
        self,
        potential_blocks: dict[
            tuple[int, ...],
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
        qudit_group: Sequence[int],
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
        in_qudits = list(q for q in qudit_group)
        stopped_cycles = {q: circuit.num_cycles for q in qudit_group}
        op_list: list[Operation] = []

        iter = self.FastRegionIterator(qudit_group, starting_cycles, circuit)
        for cycle, op in iter:
            if any(op_q not in in_qudits for op_q in op.location):
                if op.num_qudits > self.block_size:
                    raise RuntimeError(
                        'ScanPartitioner cannot handle gates larger than'
                        ' block size. You may want to use the '
                        'QuickPartitioner.',
                    )

                for op_q in op.location:
                    if op_q in in_qudits:
                        stopped_cycles[op_q] = cycle
                        in_qudits.remove(op_q)
            else:
                op_list.append(op)

            if len(in_qudits) == 0:
                break

        return (
            CircuitRegion({
                q: (starting_cycles[i], stopped_cycles[q] - 1)
                for i, q in enumerate(qudit_group)
                if stopped_cycles[q] - 1 >= starting_cycles[i]
            }),
            op_list,
        )

    class FastRegionIterator(Iterator[Tuple[int, Operation]]):
        """A circuit iterator designed to be efficient for the ScanPartitioner's
        use case."""

        def __init__(
            self,
            qudits: Sequence[int],
            starting_cycles: Sequence[int],
            circuit: Circuit,
        ) -> None:
            """Construct a FastRegionIterator."""
            self.qudits = qudits
            self.starting_cycles = {
                q: s for q, s in zip(
                    qudits, starting_cycles,
                )
            }
            self.circuit = circuit
            self.num_cycles = circuit.num_cycles

            self.inactive = list(
                sorted(zip(qudits, starting_cycles), key=lambda x: x[1]),
            )
            self.active = []
            min_cycle = self.inactive[0][1]
            for qudit, starting_cycle in self.inactive:
                if starting_cycle == min_cycle:
                    self.active.append(qudit)
            for qudit in self.active:
                self.inactive.pop(0)

            self.cycle = self.starting_cycles[self.active[0]]
            self.qudit_index = 0
            self.qudit = self.active[self.qudit_index]
            self.qudits_to_skip: set[int] = set()

        def __iter__(self) -> Iterator[tuple[int, Operation]]:
            return self

        def step(self) -> None:
            """Move the iterator one forward."""
            self.qudit_index += 1

            if self.qudit_index >= len(self.active):
                self.qudit_index = 0
                self.cycle += 1
                self.qudits_to_skip.clear()
                while True:
                    if len(self.inactive) == 0:
                        break
                    if self.inactive[0][1] != self.cycle:
                        break
                    self.active.append(self.inactive[0][0])
                    self.inactive.pop(0)

            self.qudit = self.active[self.qudit_index]

        def __next__(self) -> tuple[int, Operation]:
            """Get the next operation."""
            if self.cycle >= self.num_cycles:
                raise StopIteration
            op = self.circuit._circuit[self.cycle][self.qudit]
            while op is None:
                self.step()
                if self.cycle >= self.num_cycles:
                    raise StopIteration
                op = self.circuit._circuit[self.cycle][self.qudit]

            self.qudits_to_skip.update(op.location)
            cycle_to_return = self.cycle
            self.step()
            return cycle_to_return, op
