"""This module defines the ScanPartitioner pass."""
from __future__ import annotations

import heapq
import logging
from typing import Any
from typing import Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.machine import MachineModel
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.region import CircuitRegion
from bqskit.utils.typing import is_integer

_logger = logging.getLogger(__name__)


class ScanPartitioner(BasePass):
    """
    The HeapScanPartitioner Pass.

    This pass forms partitions in the circuit by scanning from left to right.
    This is based on the partitioner from QGo. Improves upon the ScanPart-
    tioner by using a heap to keep track of scores, and only rescores blocks
    if they were changed in the last iteration.

    References:
        Wu, Xin-Chuan, et al. "QGo: Scalable Quantum Circuit Optimization
        Using Automated Synthesis." arXiv preprint arXiv:2012.09835 (2020).
    """

    def __init__(
        self,
        block_size: int = 3,
        single_gate_score: int = 1,
        multi_gate_score: int = 1000,  # TODO: Pass callable scoring_fn instead
    ) -> None:
        """
        Construct a ScanPartitioner.

        Args:
            block_size (int): Maximum size of partitioned blocks.
                (Default: 3)

            single_gate_score (int): When evaluating potential blocks,
                use this number to score included single-qudit gates.
                (Default: 1)

            multi_gate_score (int): When evaluating potential blocks,
                use this number to score included multi-qudit gates.
                (Default: 1000)

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

        if not is_integer(single_gate_score):
            raise TypeError(
                'Expected integer for single_gate_score, '
                f'got {type(single_gate_score)}.',
            )

        if not is_integer(multi_gate_score):
            raise TypeError(
                'Expected integer for multi_gate_score, '
                f'got {type(multi_gate_score)}.',
            )

        self.block_size = block_size
        self.single_gate_score = single_gate_score
        self.multi_gate_score = multi_gate_score

    def _form_region_and_score(
        self,
        qudit_group: Sequence[int],
        divider: Sequence[int],
        circuit: Circuit,
    ) -> tuple[CircuitRegion, int]:
        """
        Find the region for the given qudit group and its score.

        Args:
            qudit_group (Sequence[int]): The qudit_group to score.

            num_cycles (int): number of cycles in the circuit.

            ops_and_cycles (Sequence): all operations and cycles in the circuit.

            divider (Sequence): maintains state of partitioning.

        Returns:
            region (CircuitRegion): region for the given qudit_group.

            score (int): score for the region.
        """
        num_cycles = circuit.get_num_cycles()

        # Move past/skip any gates that are larger than the block size
        qudits_to_increment: list[int] = []
        for qudit, cycle in enumerate(divider):
            if qudit in qudits_to_increment or cycle >= num_cycles:
                continue

            if not circuit.is_point_idle((cycle, qudit)):
                op = circuit[cycle, qudit]
                if len(op.location) > self.block_size:
                    if all(divider[q] == cycle for q in op.location):
                        qudits_to_increment.extend(op.location)
                        _logger.warning(
                            'Skipping gate larger than block size.',
                        )

        # Make sure the too-large region will be chosen
        if len(qudits_to_increment) > 0:
            region = CircuitRegion({
                qudit: (divider[qudit], divider[qudit])
                for qudit in qudits_to_increment
            })
            score = self.multi_gate_score * len(qudits_to_increment)
            score *= num_cycles

            return (region, score)

        ops_and_cycles = circuit.operations_with_cycles(
            qudits_or_region=CircuitRegion({
                qudit_index: (divider[qudit_index], num_cycles)
                for qudit_index in qudit_group
            }),
        )

        in_qudits = list(q for q in qudit_group)
        stopped_cycles = {q: num_cycles for q in qudit_group}
        score = 0

        for cycle, op in ops_and_cycles:
            if len(op.location.union(in_qudits)) != len(in_qudits):
                for qudit_index in op.location.intersection(in_qudits):
                    stopped_cycles[qudit_index] = cycle
                    in_qudits.remove(qudit_index)
            else:
                if len(op.location) > 1:
                    score += self.multi_gate_score
                else:
                    score += self.single_gate_score
            if len(in_qudits) == 0:
                break
        region = CircuitRegion({
            qudit: (divider[qudit], stopped_cycles[qudit] - 1)
            for qudit in qudit_group
        })

        return (region, score)

    def run(self, circuit: Circuit, data: dict[str, Any]) -> None:
        """
        Partition gates in a circuit into a series of CircuitGates.

        Args:
            circuit (Circuit): Circuit to be partitioned.

            data (dict[str,Any]): Optional data unique to specific run.
        """

        if self.block_size > circuit.get_size():
            _logger.warning(
                'Configured block size is greater than circuit size; '
                'blocking entire circuit.',
            )
            circuit.fold({
                qudit_index: (0, circuit.get_num_cycles())
                for qudit_index in range(circuit.get_size())
            })
            return

        # If a MachineModel is provided in the data dict, it will be used.
        # Otherwise all-to-all connectivity is assumed.
        model = None

        if 'machine_model' in data:
            model = data['machine_model']

        if (
            not isinstance(model, MachineModel)
            or model.num_qudits < circuit.get_size()
        ):
            _logger.warning(
                'MachineModel not specified or invalid;'
                ' defaulting to all-to-all.',
            )
            model = MachineModel(circuit.get_size())

        # Find all connected, `block_size`-sized groups of qudits
        # NOTE: This assumes circuit and topology qudit numbers are equal
        qudit_groups = model.get_locations(self.block_size)
        # Prune unused qudit groups
        used_qudits = [
            q for q in range(circuit.get_size())
            if not circuit.is_qudit_idle(q)
        ]
        for qudit_group in qudit_groups:
            if all([q not in used_qudits for q in qudit_group]):
                qudit_groups.remove(qudit_group)

        # divider splits the circuit into partitioned and unpartitioned spaces.
        active_qudits = circuit.get_active_qudits()
        num_cycles = circuit.get_num_cycles()
        divider = [
            0 if q in active_qudits else num_cycles
            for q in range(circuit.get_size())
        ]

        # Create the block_list, member_dict, and score_heap
        # `block_list` maintains the current region and score fore each
        # `qudit_group`.
        block_list: list[Block]
        block_list = []
        # `member_dict` has key: an qudit in the Circuit and value: a list of
        # indices into the `block_list` whose `qudit_group` contains the qudit
        # key.
        member_dict: dict[int, list[int]]
        member_dict = {q: [] for q in range(circuit.size)}
        # `score_heap` is a "max" heap that keeps track of the best scores, and
        # the index into the `block_list` that has that score.
        score_heap: list[tuple[int, int]]
        score_heap = []

        # TODO: Support avoiding operations that are too big to partition.
        for group_index, qudit_group in enumerate(qudit_groups):
            # Find the region and the score
            (region, score) = self._form_region_and_score(
                qudit_group, divider, circuit,
            )

            # Update the data structures
            block_list.append(Block(qudit_group, region, score))
            heapq.heappush(score_heap, (-1 * score, group_index))
            # Use the region's keys incase there is a too-large gate
            for qudit in region.keys():
                member_dict[qudit].append(group_index)

        # Form regions until there are no more gates to partition
        regions: list[CircuitRegion] = []
        while any(cycle < num_cycles for cycle in divider):
            # Skip any idle qudit-cycles
            amount_to_add_to_each_qudit = [0 for _ in range(circuit.get_size())]
            for qudit, cycle in enumerate(divider):
                while (
                    cycle < num_cycles
                    and circuit.is_point_idle((cycle, qudit))
                ):
                    amount_to_add_to_each_qudit[qudit] += 1
                    cycle += 1

            for qudit, amount in enumerate(amount_to_add_to_each_qudit):
                divider[qudit] += amount

            # Heap partitioning
            # Get best (up to date) block
            while len(score_heap) > 0:
                (best_score, group_index) = heapq.heappop(score_heap)
                best_score *= -1
                if best_score == block_list[group_index].score:
                    best_region = block_list[group_index].region
                    break

            if best_score is None or best_region is None:
                raise RuntimeError('No valid block found.')

            _logger.info('Found block with score: %d.' % (best_score))
            regions.append(best_region)

            # Update divider, find blocks to rescore
            rescore_set = []
            for qudit_index in best_region:
                divider[qudit_index] = best_region[qudit_index].upper + 1
                rescore_set.extend(member_dict[qudit_index])

            rescore_set = list(set(rescore_set))

            for group_index in rescore_set:
                # Find the new region and score
                (new_region, new_score) = self._form_region_and_score(
                    block_list[group_index].qudit_group,
                    divider,
                    circuit,
                )
                # Update the block_list
                block_list[group_index].region = new_region
                block_list[group_index].score = new_score
                # push to heap
                heapq.heappush(score_heap, (-1 * new_score, group_index))

        # Fold the circuit
        folded_circuit = Circuit(circuit.get_size(), circuit.get_radixes())
        # Option to keep a block's idle qudits as part of the CircuitGate
        if 'keep_idle_qudits' in data and data['keep_idle_qudits'] is True:
            for region in regions:
                small_region = circuit.downsize_region(region)
                cgc = circuit.get_slice(small_region.points)
                if len(region.location) > len(small_region.location):
                    for i in range(len(region.location)):
                        if region.location[i] not in small_region.location:
                            cgc.insert_qudit(i)
                folded_circuit.append_gate(
                    CircuitGate(cgc, True),
                    sorted(list(region.keys())),
                    list(cgc.get_params()),
                )
        else:
            for region in regions:
                region = circuit.downsize_region(region)
                if 0 < len(region) <= self.block_size:
                    cgc = circuit.get_slice(region.points)
                    folded_circuit.append_gate(
                        CircuitGate(cgc, True),
                        sorted(list(region.keys())),
                        list(cgc.get_params()),
                    )
                else:
                    folded_circuit.extend(circuit[region])
        circuit.become(folded_circuit)


class Block:
    def __init__(
        self,
        qudit_group: Sequence[int],
        region: CircuitRegion,
        score: int,
    ) -> None:
        self.qudit_group = qudit_group
        self.region = region
        self.score = score
