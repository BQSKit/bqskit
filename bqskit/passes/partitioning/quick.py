"""This module defines the QuickPartitioner pass."""
from __future__ import annotations

import logging
from typing import cast
from typing import Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.barrier import BarrierPlaceholder
from bqskit.ir.gates.circuitgate import CircuitGate
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

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """
        Partition gates in a circuit into a series of CircuitGates.

        Args:
            circuit (Circuit): Circuit to be partitioned.

            data (dict[str,Any]): Optional data unique to specific run.
        """
        # The partitioned circuit that will be built and returned
        partitioned_circuit = Circuit(circuit.num_qudits, circuit.radixes)

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
            for q in loc:
                if q in bin.active_qudits:
                    bin.active_qudits.remove(q)
                    bin.ends[q] = cycle - 1

                if active_bins[q] == bin:
                    active_bins[q] = None

            # Check if the bin is completely inactive now
            if len(bin.active_qudits) == 0:
                pending_bins.append(bin)
                return True

            return False

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
                        merging = not isinstance(bin, BarrierBin)
                        while merging:
                            merging = False
                            for p in partitioned_circuit.rear:
                                op = partitioned_circuit[p]
                                if isinstance(op.gate, BarrierPlaceholder):
                                    # Don't merge through barriers
                                    continue
                                qudits = list(op.location)

                                # if qudits is subset of loc
                                if all(q in loc for q in qudits):
                                    prev_op = partitioned_circuit.pop(p)
                                    pg = cast(CircuitGate, prev_op.gate)
                                    prev_circ = pg._circuit.copy()
                                    local_loc = [loc.index(q) for q in qudits]
                                    subc.insert_circuit(0, prev_circ, local_loc)

                                    # retry merging
                                    merging = True
                                    break

                                # if loc is a subset of qudits
                                if all(q in qudits for q in loc):
                                    prev_op = partitioned_circuit.pop(p)
                                    pg = cast(CircuitGate, prev_op.gate)
                                    prev_circ = pg._circuit.copy()
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
                            not isinstance(bin, BarrierBin),
                            True,
                        )
                        for qudit in bin.qudits:
                            if bin.ends[qudit] is not None:
                                dividing_line[qudit] = bin.ends[qudit] + 1
                            else:
                                dividing_line[qudit] = circuit.num_cycles

                        need_to_reprocess = True
                        break

                for bin in to_remove:
                    pending_bins.remove(bin)

        # Main loop
        for cycle, op in circuit.operations_with_cycles():
            point = CircuitPoint(cycle, op.location[0])
            location = op.location

            # Get all currently active bins that share at least one qudit
            overlapping_bins: list[Bin] = list({
                active_bins[q] for q in location  # type: ignore
                if active_bins[q] is not None
            })

            # Barriers close all overlapping bins
            if isinstance(op.gate, BarrierPlaceholder):
                for bin in overlapping_bins:
                    if close_bin_qudits(bin, location, cycle):
                        num_closed += 1
                    else:
                        indirect = [q for q in location if q not in bin.qudits]
                        bin.blocked_qudits.update(indirect)

                # Track the barrier to restore it in partitioned circuit
                pending_bins.append(BarrierBin(point, location, circuit))
                continue

            # Get all the currently active bins that can have op added to them
            admissible_bins = [
                bin for bin in overlapping_bins
                if bin.can_accommodate(location, self.block_size)
            ]

            # Close location on inadmissible overlapping bins
            for bin in overlapping_bins:
                if bin not in admissible_bins:
                    if close_bin_qudits(bin, location, cycle):
                        num_closed += 1

            # Select bin or make new one
            if len(admissible_bins) == 0:
                # If we cannot add this op to any bin, make a new one
                assert all(active_bins[q] is None for q in location)
                selected_bin = Bin()

            else:
                # Otherwise select an admissible bin
                found = False
                for bin in admissible_bins:
                    if all(q in bin.qudits for q in location):
                        selected_bin = bin
                        found = True
                        break

                if not found:
                    selected_bin = admissible_bins[0]

                # Close the overlapping qudits on the other admissible bins
                for bin in admissible_bins:
                    if bin != selected_bin:
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
            for active_bin in active_bins:
                if active_bin is None:
                    continue
                if active_bin == selected_bin:
                    continue

                indirect = active_bin.blocked_qudits
                indirect = indirect.union(active_bin.qudits)
                indirect = indirect.intersection(selected_bin.qudits)
                if len(indirect) != 0:
                    active_bin.blocked_qudits.update(selected_bin.qudits)
                    blockedqs = selected_bin.blocked_qudits
                    active_bin.blocked_qudits.update(blockedqs)

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

        if len(pending_bins) != 0:
            raise RuntimeError(
                'Unable to process all pending bins during partitioning.\n'
                'This should never happen and is a major issue'
                ', please make a bug report containing the input circuit.',
            )

        # Become partitioned circuit
        circuit.become(partitioned_circuit, False)


class Bin:
    """A Bin is where gates go as the QuickPartitioner sweeps a circuit."""

    id: int = 0
    """Unique ID counter for Bin instances."""

    def __init__(self) -> None:
        """Can start a new bin from an operation."""

        # The qudits in the bin
        self.qudits: list[int] = []

        # The starting cycles for each qudit (inclusive)
        self.starts: dict[int, int] = {}

        # The ending cycles for each qudit (inclusive)
        self.ends: dict[int, int | None] = {}

        # The qudits that can still accept new gates
        self.active_qudits: list[int] = []

        # Qudits that cannot be added to the bin
        self.blocked_qudits: set[int] = set()

        # Points for each operation in this bin
        self.op_list: list[CircuitPoint] = []

        self.id = Bin.id
        Bin.id += 1

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Bin) and self.id == other.id

    def __repr__(self) -> str:
        return 'Bin ' + str(self.id)

    def add_op(self, point: CircuitPoint, location: CircuitLocation) -> None:
        """Add an operation the bin."""
        for q in location:
            if q not in self.qudits:
                self.qudits.append(q)
                self.active_qudits.append(q)
                self.starts[q] = point.cycle
                self.ends[q] = None
        self.op_list.append(point)

    def can_accommodate(self, loc: CircuitLocation, block_size: int) -> bool:
        """
        Return true if the op can be added to this bin.

        An op can be added to the bin if all overlapping qudits are active in
        the bin and if the new bin won't be too large.
        """
        if any(
            q in self.blocked_qudits
            and q not in self.active_qudits
            for q in loc
        ):
            return False

        overlapping_qudits_are_active = all(
            q not in self.qudits or q in self.active_qudits
            for q in loc
        )

        size_limit = max(block_size, len(self.qudits))
        too_big = len(set(self.qudits + list(loc))) > size_limit

        return overlapping_qudits_are_active and not too_big


class BarrierBin(Bin):
    """A special bin made to mark and preserve barrier location."""

    def __init__(
        self,
        point: CircuitPoint,
        location: CircuitLocation,
        circuit: Circuit,
    ) -> None:
        """Initialize a BarrierBin with the point and location of a barrier."""
        super().__init__()

        # Add the barrier
        self.add_op(point, location)

        # Barriar bins fill the volume to the next gates

        nexts = circuit.next(point)
        ends: dict[int, int | None] = {q: None for q in location}
        for p in nexts:
            loc = circuit[p].location
            for q in loc:
                if q in ends and (ends[q] is None or ends[q] >= p.cycle):  # type: ignore # noqa # short-circuit safety for >=
                    ends[q] = p.cycle - 1

        self.ends = ends

        # Close the bin
        for q in location:
            self.active_qudits.remove(q)
