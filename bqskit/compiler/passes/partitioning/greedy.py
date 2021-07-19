# type: ignore
# TODO
"""This module defines the GreedyPartitioner pass."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.machine import MachineModel
from bqskit.ir.circuit import Circuit
from bqskit.utils.typing import is_integer

_logger = logging.getLogger(__name__)


class GreedyPartitioner(BasePass):
    """
    The GreedyPartitioner Pass.

    This pass forms partitions in the circuit greedily by forming the largest
    blocks possible first.
    """

    def __init__(
        self,
        block_size: int = 3,
        single_gate_score: int = 1,
        multi_gate_score: int = 1000,  # TODO: Pass callable scoring_fn instead
    ) -> None:
        """
        Construct a GreedyPartitioner.

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

        num_cycles = circuit.get_num_cycles()
        num_qudits_groups = len(qudit_groups)

        op_cycles = [
            [
                [0] * self.block_size for q_group in qudit_groups
            ]
            for cycle in range(num_cycles)
        ]

        for cycle, op in circuit.operations_with_cycles():
            if len(op.location) > 1:
                for q_group_index, q_group in enumerate(qudit_groups):
                    if all([qudit in q_group for qudit in op.location]):
                        for qudit in op.location:
                            op_cycles[cycle][q_group_index][
                                q_group.index(qudit)
                            ] = self.multi_gate_score
                    else:
                        for qudit in op.location:
                            if qudit in q_group:
                                op_cycles[cycle][q_group_index][
                                    q_group.index(qudit)
                                ] = -1
            else:
                qudit = op.location[0]
                for q_group_index, q_group in enumerate(qudit_groups):
                    if qudit in q_group:
                        op_cycles[cycle][q_group_index][
                            q_group.index(qudit)
                        ] = self.single_gate_score

        max_blocks = []
        for q_group_index in range(num_qudits_groups):
            block_start = 0
            block_ends = [0] * self.block_size
            score = 0
            for cycle in range(num_cycles):
                if cycle:
                    for qudit in range(self.block_size):
                        if (
                            op_cycles[cycle - 1][q_group_index][qudit] == -1
                            and op_cycles[cycle][q_group_index][qudit] != -1
                        ):
                            max_blocks.append(
                                [score, block_start, block_ends, q_group_index],
                            )
                            score = 0
                            block_start = cycle
                            block_ends = [cycle + 1] * self.block_size
                            break
                for qudit in range(self.block_size):
                    if op_cycles[cycle][q_group_index][qudit] != -1:
                        block_ends[qudit] = cycle + 1
                        score += op_cycles[cycle][q_group_index][qudit]
            max_blocks.append([score, block_start, block_ends, q_group_index])

        block_id = -1
        max_blocks.sort()
        remaining_assignments = circuit.get_size() * num_cycles
        block_map = [[-1] * circuit.get_size() for cycle in range(num_cycles)]
        while remaining_assignments:

            perform_assign = False
            if len(max_blocks) == 1:
                perform_assign = True
            else:
                block_start = max_blocks[-1][1]
                block_ends = max_blocks[-1][2]
                q_group_index = max_blocks[-1][3]
                score = 0
                for cycle in range(block_start, max(block_ends)):
                    for qudit in range(self.block_size):
                        q = qudit_groups[q_group_index][qudit]
                        if (
                            cycle < block_ends[qudit]
                            and block_map[cycle][q] == -1
                        ):
                            score += 1

                if score < max_blocks[-2][0]:
                    max_blocks[-1][0] = score
                    max_blocks.sort()
                else:
                    perform_assign = True

            if perform_assign:
                block_id += 1
                block_start = max_blocks[-1][1]
                block_ends = max_blocks[-1][2]
                q_group_index = max_blocks[-1][3]
                prev_status = None
                for cycle in range(block_start, max(block_ends)):
                    status = [
                        block_map[cycle][qudit_groups[q_group_index][qudit]]
                        for qudit in range(self.block_size)
                    ]
                    if prev_status and len(prev_status) <= len(
                            status,
                    ) and status != prev_status:
                        block_id += 1
                    for qudit in range(self.block_size):
                        if (
                            cycle < block_ends[qudit]
                            and block_map[cycle][
                                qudit_groups[q_group_index][qudit]
                            ] == -1
                        ):
                            block_map[cycle][
                                qudit_groups[q_group_index]
                                [qudit]
                            ] = block_id
                            remaining_assignments -= 1
                    prev_status = status
                del max_blocks[-1]

        for cycle in range(num_cycles):
            if not cycle or block_map[cycle] == block_map[cycle - 1]:
                continue
            indices = [{}, {}]
            for i in range(2):
                for qudit in range(circuit.get_size()):
                    block = block_map[cycle - i][qudit]
                    if block not in indices[i]:
                        indices[i][block] = []
                    indices[i][block].append(qudit)
            for prev_blocks, prev_qudits in indices[1].items():
                for current_qudits in indices[0].values():
                    if all([qudit in prev_qudits for qudit in current_qudits]):
                        for qudit in current_qudits:
                            block_map[cycle][qudit] = prev_blocks

        blocks = {}
        for cycle in range(num_cycles):
            for qudit in range(circuit.get_size()):
                if block_map[cycle][qudit] not in blocks:
                    blocks[block_map[cycle][qudit]] = {}
                    blocks[block_map[cycle][qudit]][-1] = cycle
                blocks[block_map[cycle][qudit]][qudit] = cycle

        block_order = []
        for block in blocks.values():
            block_order.append([block, block[-1]])
        block_order.sort(reverse=True, key=lambda x: x[1])

        for block, start_cycle in block_order:
            points_in_block = []
            for cycle, op in circuit.operations_with_cycles():
                qudit = op.location[0]
                if (
                    qudit in block
                    and cycle >= start_cycle
                    and cycle <= block[qudit]
                ):
                    points_in_block.append((cycle, qudit))

            circuit.fold(circuit.get_region(points_in_block))
