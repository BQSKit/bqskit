"""This module defines the QuickPartitioner pass."""
from __future__ import annotations

import logging
from typing import Any
from typing import cast
from typing import Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CircuitGate
from bqskit.ir.location import CircuitLocation
from bqskit.ir.point import CircuitPoint
from bqskit.utils.typing import is_integer

_logger = logging.getLogger(__name__)


class QuickPartitioner(BasePass):
    """
    A partitioner that iterates over circuit gates only once.

    This pass forms partitions in the circuit by iterating over the operations
    in a topological order and binning them into blocks.
    """

    def __init__(self, block_size: int = 3) -> None:
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

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """
        Partition gates in a circuit into a series of CircuitGates.

        Args:
            circuit (Circuit): Circuit to be partitioned.

            data (dict[str,Any]): Optional data unique to specific run.
        """
        # The partitioned circuit that will be built and returned
        partitioned_circuit = Circuit(circuit.num_qudits, circuit.radixes)

        # If block size >= circuit size, return the circuit as a block
        if self.block_size >= circuit.num_qudits:
            _logger.debug(
                'Configured block size is greater than or equal to'
                'circuit size; blocking entire circuit.',
            )
            partitioned_circuit.append_circuit(
                circuit,
                list(range(circuit.num_qudits)),
                True,
            )
            circuit.become(partitioned_circuit)
            return

        # Tracks bins with at least one active_qudit
        active_bins: list[Bin | None] = [
            None for _ in range(circuit.num_qudits)
        ]
        # Tracks the first cycle in circuit not included in partitioned_circuit
        dividing_line: dict[int, int] = {
            i: 0 if circuit._front[i] is None else circuit._front[i].cycle  # type: ignore  # noqa
            for i in range(circuit.num_qudits)
        }

        # Inactive bins that cannot yet be put on partitioned_circuit
        pending_bins: list[Bin] = []

        # Track how many bins have been closed without processing
        num_closed = 0

        def close_bin_qudits(bin: Bin, loc: Sequence[int], cycle: int) -> bool:
            """Deactivate `qudits` in `bin`; return True if bin is inactive."""
            to_return = False
            for q in loc:
                if q in bin.active_qudits:
                    bin.active_qudits.remove(q)
                    bin.ends[q] = cycle - 1

                if active_bins[q] == bin:
                    active_bins[q] = None

            # Check if the bin is completely inactive now
            if len(bin.active_qudits) == 0:
                pending_bins.append(bin)
                to_return = True
            return to_return

        def process_pending_bins() -> None:
            """Add pending bins that can be added to the partitioned circuit."""
            need_to_reprocess = True
            while need_to_reprocess:
                need_to_reprocess = False
                to_remove = []

                for bin in pending_bins:
                    if all(
                        dividing_line[qudit] == start
                        for qudit, start in bin.starts.items()
                    ):
                        to_remove.append(bin)
                        subc = circuit.get_slice(bin.op_list)
                        loc = list(sorted(bin.qudits))

                        # Merge previously placed blocks if possible
                        merging = True
                        while merging:
                            merging = False
                            for p in partitioned_circuit.rear:
                                qudits = partitioned_circuit[p].location

                                # if qudits is subset of loc
                                if all(q in loc for q in qudits):
                                    prev_op = partitioned_circuit.pop(p)
                                    pg = cast(CircuitGate, prev_op.gate)
                                    prev_circ = pg._circuit
                                    local_loc = [loc.index(q) for q in qudits]
                                    subc.insert_circuit(0, prev_circ, local_loc)

                                    # retry merging
                                    merging = True
                                    break

                                # if loc is a subset of qudits
                                if all(q in qudits for q in loc):
                                    prev_op = partitioned_circuit.pop(p)
                                    pg = cast(CircuitGate, prev_op.gate)
                                    prev_circ = pg._circuit
                                    lloc = [qudits.index(q) for q in loc]
                                    prev_circ.append_circuit(subc, lloc)
                                    subc.become(prev_circ)
                                    loc = qudits

                                    # retry merging
                                    merging = True
                                    break

                        # Place circuit
                        partitioned_circuit.append_circuit(
                            subc,
                            loc,
                            True,
                            True,
                        )
                        for qudit in bin.qudits:
                            dividing_line[qudit] = bin.ends[qudit] + 1  # type: ignore  # noqa
                        need_to_reprocess = True

                for bin in to_remove:
                    pending_bins.remove(bin)

        # Main loop
        for cycle, op in circuit.operations_with_cycles():
            point = CircuitPoint(cycle, op.location[0])
            location = op.location

            # Get all currently active bins that share atleast one qudit
            overlapping_bins: list[Bin] = list({
                active_bins[q] for q in location  # type: ignore
                if active_bins[q] is not None
            })

            # Get all the currently active bins that can have op added to them
            admissible_bins = [
                b for b in overlapping_bins
                if b.can_accommodate(location, self.block_size)
            ]

            # Close location on inadmissible overlapping bins
            for bin in overlapping_bins:
                if bin not in admissible_bins:
                    if close_bin_qudits(bin, location, cycle):
                        num_closed += 1

            # If we cannot add this op to any bin, make a new one
            if len(admissible_bins) == 0:
                assert all(active_bins[q] is None for q in location)
                new_bin = Bin(point, location)
                for q in location:
                    active_bins[q] = new_bin

                # Block qudits to prevent circular dependencies
                for bin in overlapping_bins:
                    bin.blocked_qudits.update(new_bin.qudits)

                    for active_bin in active_bins:
                        if active_bin is None or active_bin == bin:
                            continue

                        indirect = active_bin.blocked_qudits
                        indirect = indirect.intersection(bin.qudits)
                        if len(indirect) != 0:
                            active_bin.blocked_qudits.update(new_bin.qudits)

            else:
                # Add to first admissible bin
                selected_bin = admissible_bins[0]

                # Deactivate the rest
                for bin in admissible_bins[1:]:
                    if close_bin_qudits(bin, location, cycle):
                        num_closed += 1

                # Add op to selected_bin
                selected_bin.add_op(point, location)
                for q in location:
                    if active_bins[q] is None:
                        active_bins[q] = selected_bin
                    else:
                        assert active_bins[q] == selected_bin

                # Block qudits to prevent circular dependencies
                for bin in overlapping_bins:
                    if bin == selected_bin:
                        continue
                    bin.blocked_qudits.update(selected_bin.qudits)

                    for active_bin in active_bins:
                        if active_bin is None or active_bin == bin:
                            continue

                        indirect = active_bin.blocked_qudits
                        indirect = indirect.intersection(bin.qudits)
                        if len(indirect) != 0:
                            active_bin.blocked_qudits.update(new_bin.qudits)

            # If a new bin was finalized, reprocess pending bins
            if num_closed >= 5:
                process_pending_bins()
                num_closed = 0

        # Close remaining active bins
        for b in active_bins:
            if b is not None:
                close_bin_qudits(b, b.qudits, circuit.num_cycles)

        # Process remaining bins
        process_pending_bins()

        # Become partitioned circuit
        circuit.become(partitioned_circuit, False)


class Bin:
    """A Bin is where gates go as the QuickPartitioner sweeps a circuit."""

    id: int = 0
    """Unique ID counter for Bin instances."""

    def __init__(
        self,
        point: CircuitPoint,
        location: CircuitLocation,
    ) -> None:
        """Can start a new bin from an operation."""

        # The qudits in the bin
        self.qudits: list[int] = list(location)

        # The starting cycles for each qudit (inclusive)
        self.starts: dict[int, int] = {q: point.cycle for q in location}

        # The ending cycles for each qudit (inclusive)
        self.ends: dict[int, int | None] = {q: None for q in location}

        # The qudits that can still accept new gates
        self.active_qudits: list[int] = list(location)

        # Qudits that cannot be added to the bin
        self.blocked_qudits: set[int] = set()

        # Points for each operation in this bin
        self.op_list: list[CircuitPoint] = [point]

        self.id = Bin.id
        Bin.id += 1

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Bin) and self.id == other.id

    def add_op(
        self,
        point: CircuitPoint,
        location: CircuitLocation,
    ) -> None:
        """Add an operation the bin."""
        for q in location:
            if q not in self.qudits:
                self.qudits.append(q)
                self.active_qudits.append(q)
                self.starts[q] = point.cycle
        self.op_list.append(point)

    def can_accommodate(self, loc: CircuitLocation, block_size: int) -> bool:
        """
        Return true if the op can be added to this bin.

        An op can be added to the bin if all overlapping qudits are active in
        the bin and if the new bin won't be too large.
        """
        if any(q in loc for q in self.blocked_qudits):
            return False

        overlapping_qudits_are_active = all(
            q not in self.qudits or q in self.active_qudits
            for q in loc
        )

        size_limit = max(block_size, len(self.qudits))
        too_big = len(set(self.qudits + list(loc))) > size_limit

        return overlapping_qudits_are_active and not too_big
