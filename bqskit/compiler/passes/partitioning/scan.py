"""This module defines the ScanPartitioner pass."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.machine import MachineModel
from bqskit.ir.circuit import Circuit
from bqskit.ir.region import CircuitRegion
from bqskit.utils.typing import is_integer

_logger = logging.getLogger(__name__)


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
        block_size: int = 3,
        single_gate_score: int = 1,
        multi_gate_score: int = 1000,  # TODO: Pass callable scoring_fn instead
    ) -> None:
        """
        Construct a SimplePartitioner.

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

        # divider splits the circuit into partitioned and unpartitioned spaces.
        divider = [0 for _ in range(circuit.get_size())]
        num_cycles = circuit.get_num_cycles()
        regions: list[CircuitRegion] = []

        # Do while there are still gates to partition
        while any(cycle < num_cycles for cycle in divider):

            # Find the scores of the qudit groups.
            best_score = None
            best_region = None

            for qudit_group in qudit_groups:

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
                            stopped_cycles[qudit_index] = cycle - 1
                            in_qudits.remove(qudit_index)
                    else:
                        if len(op.location) > 1:
                            score += self.multi_gate_score
                        else:
                            score += self.single_gate_score

                    if len(in_qudits) == 0:
                        break

                if best_score is None or score > best_score:
                    best_score = score
                    best_region = CircuitRegion({
                        qudit_index: (
                            divider[qudit_index],
                            stopped_cycles[qudit_index],
                        )
                        for qudit_index in qudit_group
                    })

            if best_score is None or best_region is None:
                raise RuntimeError('No valid block found.')

            _logger.info('Found block with score: {best_score}.')
            regions.append(best_region)

            # Update divider
            for qudit_index in best_region:
                divider[qudit_index] = best_region[qudit_index].upper + 1

        circuit.batch_fold(regions)
