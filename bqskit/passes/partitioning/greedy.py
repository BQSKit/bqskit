"""This module defines the GreedyPartitioner pass."""
from __future__ import annotations

import bisect
import logging
from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.interval import CycleInterval
from bqskit.ir.region import CircuitRegion
from bqskit.utils.typing import is_integer

_logger = logging.getLogger(__name__)


class GreedyPartitioner(BasePass):
    """
    The GreedyPartitioner Pass.

    This pass partitions a circuit by forming the largest regions first.
    """

    def __init__(self, block_size: int = 3) -> None:
        """
        Construct a GreedyPartitioner.

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

        # For each gate, calculate the best region surrounding it
        total_num_gates = 0
        regions: list[CircuitRegion] = []
        all_bounds: list[list[CycleInterval]] = [
            [] for q in range(circuit.num_qudits)
        ]
        potential_regions = {}
        for cycle, op in circuit.operations_with_cycles():
            point = (cycle, op.location[0])
            if len(op.location) > self.block_size:
                regions.append(circuit.get_region([point]))
                for qudit, bounds in circuit.get_region([point]).items():
                    bisect.insort(all_bounds[qudit], bounds)
                continue
            total_num_gates += 1
            region = circuit.surround(
                point,
                self.block_size,
                fail_quickly=True,
            )

            potential_regions[point] = (len(circuit[region]), region)

        # Form regions until there are no more gates to partition
        num_partitioned_gates = 0
        while num_partitioned_gates < total_num_gates:

            # Pick largest region
            s = sorted(potential_regions.values(), key=lambda x: x[0])
            num_gates, best_region = s[-1]
            num_partitioned_gates += num_gates
            regions.append(best_region)

            # Update all_bounds
            for qudit, bounds in best_region.items():
                bisect.insort(all_bounds[qudit], bounds)

            # Update others
            to_remove = []
            to_update = []
            for point, value in potential_regions.items():
                region = value[1]

                if point in best_region or region == best_region:
                    to_remove.append(point)
                    continue

                if best_region.overlaps(region):
                    to_update.append(point)
                    continue

            for point in to_remove:
                potential_regions.pop(point)

            for point in to_update:
                # Calculate bounding region
                bounding_region = {}
                cycle = point[0]
                for qudit, bounds_list in enumerate(all_bounds):
                    # find first bound with lower larger than cycle
                    if len(bounds_list) == 0:
                        bounding_region[qudit] = (
                            0, circuit.num_cycles - 1,
                        )
                        continue

                    index_of_first_larger = None
                    for i, bounds in enumerate(bounds_list):
                        if bounds.lower > cycle:
                            index_of_first_larger = i
                            break

                    if index_of_first_larger is None:
                        bounding_region[qudit] = (
                            bounds_list[-1][1] + 1,
                            circuit.num_cycles - 1,
                        )
                    elif index_of_first_larger == 0:
                        bounding_region[qudit] = (0, bounds_list[0][0] - 1)
                    else:
                        bounding_region[qudit] = (
                            bounds_list[index_of_first_larger - 1][1] + 1,
                            bounds_list[index_of_first_larger][0] - 1,
                        )

                    if bounding_region[qudit][0] > bounding_region[qudit][1]:
                        bounding_region.pop(qudit)

                bounding_region = CircuitRegion(bounding_region)

                # Recalculate region
                region = circuit.surround(
                    point,
                    self.block_size,
                    bounding_region,
                    True,
                )

                potential_regions[point] = (len(circuit[region]), region)

        # TODO: Merge regions that can be merged together

        # Fold the circuit
        folded_circuit = Circuit(circuit.num_qudits, circuit.radixes)
        regions = self.topo_sort(regions)
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
                    list(cgc.params),
                )
        else:
            for region in regions:
                region = circuit.downsize_region(region)
                if 0 < len(region) <= self.block_size:
                    cgc = circuit.get_slice(region.points)
                    folded_circuit.append_gate(
                        CircuitGate(cgc, True),
                        sorted(list(region.keys())),
                        list(cgc.params),
                    )
                else:
                    folded_circuit.extend(circuit[region])
        circuit.become(folded_circuit)

    def topo_sort(self, regions: list[CircuitRegion]) -> list[CircuitRegion]:
        """Topologically sort regions."""
        sorted_regions: list[CircuitRegion] = []
        in_adj_list: list[list[int]] = [[] for _ in range(len(regions))]

        for i, region1 in enumerate(regions):
            for j, region2 in enumerate(regions):
                if i == j:
                    continue
                if region1.depends_on(region2):
                    in_adj_list[i].append(j)  # j points to i

        already_selected: list[int] = []
        while len(already_selected) != len(regions):
            selected = None
            for i, in_nodes in enumerate(in_adj_list):
                if i not in already_selected and len(in_nodes) == 0:
                    selected = i
                    break

            if selected is None:
                raise RuntimeError('Unable to topologically sort regions.')

            sorted_regions.append(regions[selected])
            already_selected.append(selected)
            for in_nodes in in_adj_list:
                if selected in in_nodes:
                    in_nodes.remove(selected)

        return sorted_regions
