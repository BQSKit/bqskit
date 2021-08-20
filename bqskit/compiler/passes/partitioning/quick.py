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

    def add_final_blocks(self, block, qudits, active_blocks, all_blocks):
        """
        Add blocks with all inactive qudits to the all_blocks list and
        remove them from the active_blocks list.
        
        """

        # Compile the qudits from the new operation,
        # the active qudits of the block being updated,
        # and the qudits in the block's inadmissible list
        qudits = set(qudits)
        if block:
            qudits.update([qudit for qudit in block if qudit != -1])
            qudits.update(block[-1])

        remove_blocks = []

        # For all active blocks
        for active_block in active_blocks:

            # If the active block is different than the block being updated
            if active_block != block:

                # If any of the qudits are in the active block or its inadmissible list,
                # then add those qudits to the inadmissible list of the active block
                if any([qudit in active_block or qudit in active_block[-1] for qudit in qudits]):
                    active_block[-1].update(qudits)

                # If the active block has reached its maximum size
                # and all of its qudits are inadmissible,
                # then add it to the remove list
                if len(active_block) - 1 == self.block_size and \
                   all([qudit in active_block[-1] for qudit in active_block if qudit != -1]):
                    remove_blocks.append(active_block)

        # Remove all blocks in the remove list from the active list
        # and add them to the final all blocks list after deleting
        # their inadmissible list
        for remove_block in remove_blocks:
            del remove_block[-1]
            all_blocks.append(remove_block)
            active_blocks.remove(remove_block)

    def topo_sort(self, regions):
        """
        Topologically sort circuit regions.
        
        """

        # Number of regions in the circuit
        num_regions = len(regions)
       
        # For each region, generate the number of in edges
        # and the list of all out edges
        in_edges = [0]*num_regions
        out_edges = [[] for _ in range(num_regions)]
        for i in range(num_regions-1):
            for j in range(i+1, num_regions):
                dependency = regions[i].dependency(regions[j])
                if dependency == 1:
                    in_edges[i] += 1
                    out_edges[j].append(i)
                elif dependency == -1:
                    in_edges[j] += 1
                    out_edges[i].append(j)

        # Convert the list of number of in edges in to a min-heap
        in_edges = [[num_edges, i] for i, num_edges in enumerate(in_edges)]
        heapq.heapify(in_edges)

        index = 0
        sorted_regions = []

        # While there are regions remaining to be sorted
        while index < num_regions:

            # Select the regions with zero remaining in edges
            selections = []
            while in_edges and not in_edges[0][0]:
                selections.append(heapq.heappop(in_edges))

            if not selections:
                raise RuntimeError('Unable to topologically sort regions.')

            # Add the regions to the sorted list
            for region in selections:
                sorted_regions.append(regions[region[1]])
                index += 1

            # Remove the regions from all other regions' in edges counts
            for i in range(len(in_edges)):
                in_edges[i][0] -= sum([in_edges[i][1] in out_edges[region[1]] for region in selections])

            # Convert in edges into a min-heap
            heapq.heapify(in_edges)

        return sorted_regions

    def run(self, circuit: Circuit, data: dict[str, Any]) -> None:
        """
        Partition gates in a circuit into a series of CircuitGates.

        Args:
            circuit (Circuit): Circuit to be partitioned.

            data (dict[str,Any]): Optional data unique to specific run.
        """

        # If block size > circuit size, return the circuit as a block
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

        # List to hold the final blocks
        all_blocks = []

        # List to hold the active blocks
        active_blocks = []

        # For each cycle, operation in topological order
        for cycle, op in circuit.operations_with_cycles():
            
            # Get the qudits of the operation
            qudits = op.location._location

            # Compile a list of admissible blocks out of the
            # active blocks for the operation
            admissible_blocks = []
            for index, block in enumerate(active_blocks):
                if all([qudit not in block[-1] for qudit in qudits]):
                    admissible_blocks.append(index)

            # Boolean indicator to capture if an active block
            # has been found for the operation
            found = False

            # For all admissible blocks, check if all operation
            # qudits are in the block. If such a block is found,
            # update the upper region bound for the corresponding
            # qudits, and raise the found boolean
            for block in [active_blocks[index] for index in admissible_blocks]:
                if all([qudit in block for qudit in qudits]):
                    for qudit in qudits:
                        block[qudit][1] = cycle
                    found = True
                    break

            # If such a block is not found
            if not found:

                # For all admissible blocks, check if the operation
                # qudits can be added to the block without breaching
                # the size limit. If such a block is found, add the
                # new qudits, update the region bounds, check if any
                # blocks are finalized, and raise the found boolean
                for block in [active_blocks[index] for index in admissible_blocks]:
                    if len(set(list(qudits) + list(block.keys()))) - 1 <= self.block_size:
                        for qudit in qudits:
                            if qudit not in block:
                                block[qudit] = [cycle, cycle]
                            else:
                                block[qudit][1] = cycle
                        self.add_final_blocks(block, qudits, active_blocks, all_blocks)
                        found = True
                        break

            # If a block is still not found, check if any blocks are finalized
            # with the new operation qudits, create a new block, and add it
            # to the list of active blocks
            if not found:
                self.add_final_blocks(None, qudits, active_blocks, all_blocks)
                block = {qudit: [cycle, cycle] for qudit in qudits}
                block[-1] = set()
                active_blocks.append(block)

        # Convert all remaining active blocks at the end
        # of the circuit into final blocks
        for block in active_blocks:
            del block[-1]
            all_blocks.append(block)

        # Convert all the final blocks into circuit regions
        # and topologically sort them
        regions = []
        for block in all_blocks:
            region = CircuitRegion({qudit: (block[qudit][0], block[qudit][1]) for qudit in block})
            regions.append(region)
        regions = self.topo_sort(regions)

        # Fold the regions into a new circuit
        folded_circuit = Circuit(circuit.get_size(), circuit.get_radixes())
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

        # Copy the new circuit to the original circuit
        circuit.become(folded_circuit)
