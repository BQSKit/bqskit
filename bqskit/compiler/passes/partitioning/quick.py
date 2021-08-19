"""This module defines the QuickPartitioner pass."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.machine import MachineModel
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.region import CircuitRegion
from bqskit.utils.typing import is_integer

_logger = logging.getLogger(__name__)

class QuickPartitioner(BasePass):
    """
    The QuickPartitioner Pass.

    This pass forms partitions in the circuit by iterating over the
    operations in a topological order and binning them into blocks.

    """

    def __init__(
        self,
        block_size: int = 3,
    ) -> None:
        """
        Construct a QuickPartitioner.

        Args:
            block_size (int): Maximum size of partitioned blocks.
                (Default: 3)

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

    def check_intersecting_partitions(self, block, qudits, cycle, all_blocks):
        """
        Check if adding an operation to a block with make it intersect with
        other blocks (bounds of qudits intersecting).
        
        """
        
        for all_block in all_blocks:
            
            if all_block == block:
                continue

            inter = [qudit for qudit in [q for q in block.keys() if q != -1] + list(qudits) if qudit in all_block]

            if not inter:
                continue

            depend = []
            for qudit in inter:
                if qudit in block:
                    depend.append(all_block[qudit] < block[qudit])
                else:
                    depend.append(all_block[qudit] < [cycle, cycle])

            if True in depend and False in depend:
                return False

        return True

    def add_final_blocks(self, block, qudits, active_blocks, all_blocks):
        """
        Add blocks with all inactive qudits to the all_blocks list and
        remove them from the active_blocks list.
        
        """

        remove_blocks = []
        for other_block in active_blocks:
            if other_block != block:
                for qudit in qudits:
                    if qudit in other_block and qudit not in other_block[-1]:
                        other_block[-1].append(qudit)
                if len(other_block[-1]) == self.block_size:
                    remove_blocks.append(other_block)

        for remove_block in remove_blocks:
            del remove_block[-1]
            all_blocks.append(remove_block)
            active_blocks.remove(remove_block)

    def topo_sort(self, regions):
        """
        Topologically sort circuit regions.
        
        """

        num_regions = len(regions)
       
        in_adj_list = [[] for _ in range(num_regions)]

        for i, region1 in enumerate(regions):
            for j, region2 in enumerate(regions):
                if i == j:
                    continue
                if region1.depends_on(region2):
                    in_adj_list[j].append(i)

        index = 0
        sorted_regions = []
        already_selected = []
        while index < num_regions:

            selected = None
            for i, in_nodes in enumerate(in_adj_list):
                if i not in already_selected and not in_nodes:
                    selected = i
                    break

            if selected is None:
                raise RuntimeError('Unable to topologically sort regions.')

            sorted_regions.append(regions[selected])
            already_selected.append(selected)

            for in_nodes in in_adj_list:
                if selected in in_nodes:
                    in_nodes.remove(selected)

            index += 1

        return sorted_regions

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

        all_blocks = []
        active_blocks = []
        for cycle, op in circuit.operations_with_cycles():
            
            qudits = op.location._location
            found = False
            admissible_blocks = []
            for index, block in enumerate(active_blocks):
                if all([qudit not in block[-1] for qudit in qudits]):
                    admissible_blocks.append(index)
            for block in [active_blocks[index] for index in admissible_blocks]:
                if all([qudit in block for qudit in qudits]):
                    for qudit in qudits:
                        block[qudit][1] = cycle
                    found = True
                    break
            if not found:
                for block in [active_blocks[index] for index in admissible_blocks]:
                    #print(qudits, block.keys(), set(list(qudits) + list(block.keys())))
                    if len(set(list(qudits) + list(block.keys()))) - 1 <= self.block_size \
                       and self.check_intersecting_partitions(block, qudits, cycle, active_blocks):
                        for qudit in qudits:
                            if qudit not in block:
                                block[qudit] = [cycle, cycle]
                            else:
                                block[qudit][1] = cycle
                        self.add_final_blocks(block, qudits, active_blocks, all_blocks)
                        found = True
                        break
            if not found:
                self.add_final_blocks(None, qudits, active_blocks, all_blocks)
                block = {qudit: [cycle, cycle] for qudit in qudits}
                block[-1] = []
                active_blocks.append(block)
        for block in active_blocks:
            del block[-1]
            all_blocks.append(block)

        regions = []
        for block in all_blocks:
            region = CircuitRegion({qudit: (block[qudit][0], block[qudit][1]) for qudit in block})
            regions.append(region)
        regions = self.topo_sort(regions)

        folded_circuit = Circuit(circuit.get_size(), circuit.get_radixes())
        for region in regions:
            #print(region)
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
