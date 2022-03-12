"""This module implements the CircuitIterator class."""
from __future__ import annotations

from typing import Iterator
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

from bqskit.ir.operation import Operation
from bqskit.ir.point import CircuitPoint
from bqskit.ir.point import CircuitPointLike
from bqskit.ir.region import CircuitRegion
from bqskit.ir.region import CircuitRegionLike
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_sequence

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit


class CircuitIterator(
    Iterator[
        Union[
            Operation,
            Tuple[int, Operation],  # if and_cycles == True
        ]
    ],
):
    """
    The CircuitIterator Class.

    A CircuitIterator can iterate through a circuit in a few different ways. By
    default it can iterate through all operations in the circuit in simulation
    order. Additionally, it can iterate all the operations on a qudit or set of
    qudits or iterate through a specified CircuitRegion. If iterating all
    operations in a region or on some qudits, you can choose to exclude
    operations that only are partially in the specified area.
    """

    def __init__(
            self,
            circuit: Circuit,
            start: CircuitPointLike = CircuitPoint(0, 0),
            end: CircuitPointLike | None = None,
            qudits_or_region: CircuitRegionLike | Sequence[int] | None = None,
            exclude: bool = False,
            reverse: bool = False,
            and_cycles: bool = False,
    ) -> None:
        """
        Construct a CircuitIterator.

        Args:
            circuit (Circuit): The circuit to iterate through.

            start (CircuitPointLike): Only iterate through points greater
                than or equal to `start`. Defaults to start at the beginning
                of the circuit. (Default: (0, 0))

            end (CircuitPointLike | None): Only iterate through points
                less than or equal to this. If left as None, iterates
                until the end of the circuit. (Default: None)

            qudits_or_region (CircuitRegionLike | Sequence[int] | None):
                Determines the way the circuit is iterated. If a region
                is given, then iterate through operations in the region.
                If a sequence of qudit indices is given, then only iterate
                the operations touching those qudits. If left as None,
                then iterate through the entire circuit in simulation order.
                (Default: None)

            exclude (bool): If iterating through a region or only some
                qudits and `exclude` is true, then do not yield operations
                that are only partially in the region or on the desired
                qudits. This may result in a sequence of operations that
                does not occur in simulation order in the circuit.
                (Default: False)

            reverse (bool): Reverse the ordering. If true, then end acts
                as start and vice versa. (Default: False)

            and_cycles (bool): If true, in addition to the operation,
                return the cycle index where it was found. (Default: False)
        """
        if not CircuitPoint.is_point(start):
            raise TypeError(f'Expected point for start, got {type(start)}.')

        if end is not None and not CircuitPoint.is_point(end):
            raise TypeError(f'Expected point for end, got {type(end)}.')

        if end is None:
            end = CircuitPoint(
                circuit.num_cycles - 1,
                circuit.num_qudits - 1,
            )

        self.circuit = circuit
        self.start = CircuitPoint(*start)
        self.end = CircuitPoint(*end)
        self.exclude = exclude
        self.reverse = reverse
        self.and_cycles = and_cycles

        # Set mode of iteration:
        if qudits_or_region is None:
            # iterate through the entire circuit normally
            self.qudits = list(range(self.circuit.num_qudits))
            self.region = CircuitRegion({
                qudit: (0, self.circuit.num_cycles)
                for qudit in self.qudits
            })

        elif CircuitRegion.is_region(qudits_or_region):
            # iterate through the region in the circuit
            self.qudits = list(qudits_or_region.keys())
            self.region = CircuitRegion(qudits_or_region)

        elif is_sequence(qudits_or_region):
            # iterate through the circuit but only on the specified qudits
            if not all(is_integer(qudit) for qudit in qudits_or_region):
                raise TypeError('Expected region or sequence of indices.')

            if not all(
                0 <= qudit < self.circuit.num_qudits
                for qudit in qudits_or_region
            ):
                raise ValueError('Invalid sequence of qudit indices.')

            self.qudits = list(qudits_or_region)
            self.region = CircuitRegion({
                qudit: (0, self.circuit.num_cycles)
                for qudit in self.qudits
            })

        self.max_qudit = max(self.qudits)
        self.min_qudit = min(self.qudits)
        self.min_cycle = self.region.min_cycle
        self.max_cycle = self.region.max_cycle

        if start < (self.min_cycle, self.min_qudit):
            start = CircuitPoint(self.min_cycle, self.min_qudit)

        if end > (self.max_cycle, self.max_qudit):
            end = CircuitPoint(self.max_cycle, self.max_qudit)

        assert isinstance(start, CircuitPoint)  # TODO: Typeguard
        assert isinstance(end, CircuitPoint)  # TODO: Typeguard

        # Pointer into the circuit structure
        self.cycle = start.cycle if not self.reverse else end.cycle
        self.qudit = start.qudit if not self.reverse else end.qudit

        # Used to track changes to circuit structure
        self.num_ops = self.circuit.num_operations
        self.num_cycles = self.circuit.num_cycles
        self.num_qudits = self.circuit.num_qudits

        # Ensure operations are only returned once
        self.qudits_to_skip: set[int] = set()

    def increment_iter(self) -> None:
        """Increment the iterator to the next valid circuit point."""
        while (
            self.qudit in self.qudits_to_skip
            or self.qudit not in self.qudits
            or (
                self.cycle not in self.region[self.qudit]
                and self.cycle <= self.max_cycle
            )
        ):
            self.qudit += 1

            if self.qudit > self.max_qudit:
                self.qudit = self.min_qudit
                self.cycle += 1
                self.qudits_to_skip.clear()

    def decrement_iter(self) -> None:
        """Decrement the iterator to the next valid circuit point."""
        while (
            self.qudit in self.qudits_to_skip
            or self.qudit not in self.qudits
            or (
                self.cycle not in self.region[self.qudit]
                and self.cycle >= self.min_cycle
            )
        ):
            self.qudit -= 1

            if self.qudit < self.min_qudit:
                self.qudit = self.max_qudit
                self.cycle -= 1
                self.qudits_to_skip.clear()

    def step(self) -> None:
        """Move the iterator one step."""
        if not self.reverse:
            self.increment_iter()
        else:
            self.decrement_iter()

        point = (self.cycle, self.qudit)
        if point < self.start or point > self.end:
            raise StopIteration

    def __next__(self) -> Operation | tuple[int, Operation]:
        if (
            self.num_ops != self.circuit.num_operations
            or self.num_cycles != self.circuit.num_cycles
            or self.num_qudits != self.circuit.num_qudits
        ):
            raise RuntimeError('Circuit changed under iteration.')

        while True:
            self.step()
            op = self.circuit._circuit[self.cycle][self.qudit]

            if op is None:
                self.qudits_to_skip.add(self.qudit)
                continue

            if self.exclude:
                if not all(qudit in self.qudits for qudit in op.location):
                    continue

                if not all(
                    self.region.overlaps((self.cycle, qudit))
                    for qudit in op.location
                ):
                    continue

            self.qudits_to_skip.update(op.location)

            if self.and_cycles:
                return self.cycle, op

            return op

    def __iter__(self) -> CircuitIterator:
        return self
