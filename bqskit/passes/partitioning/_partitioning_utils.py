from __future__ import annotations

import bisect
import heapq
from typing import Iterable
from typing import Iterator
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar

from bqskit.ir.circuit import Circuit
from bqskit.ir.operation import Operation
# Generic utils

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison

T = TypeVar('T', bound='SupportsRichComparison')


class PriorityQueueSet(Set[T]):
    """
    Source: https://stackoverflow.com/questions/407734/
        a-generic-priority-queue-for-python
    Combined priority queue and set data structure.

    Acts like a priority queue, except that its items are guaranteed to be
    unique. Provides O(1) membership test, O(log N) insertion and O(log N)
    removal of the smallest item.

    Important: the items of this data structure must be both comparable and
    hashable (i.e. must implement __cmp__ and __hash__). This is true of
    Python's built-in objects, but you should implement those methods if you
    want to use the data structure for custom objects.
    """

    def __init__(self, items: Iterable[T] = tuple()) -> None:
        """
        Create a new PriorityQueueSet.

        Arguments:
            items (list): An initial item list - it can be unsorted and
                non-unique. The data structure will be created in O(N).
        """
        self.set: set[T] = set(items)
        self.heap: list[T] = list(items)
        heapq.heapify(self.heap)

    def __contains__(self, item: object) -> bool:
        return item in self.set

    def __len__(self) -> int:
        return len(self.set)

    def pop(self) -> T:
        """Remove and return the smallest item from the queue."""
        smallest = heapq.heappop(self.heap)
        self.set.remove(smallest)
        return smallest

    def push(self, item: T) -> None:
        """Add ``item`` to the queue if doesn't already exist."""
        if item not in self.set:
            self.set.add(item)
            heapq.heappush(self.heap, item)


# BQSKit utils


class SimpleCircuitPoint(Tuple[int, int]):
    """A simple, more efficient version of BQSKit's CircuitPoint."""
    def __new__(
        cls,
        cycle: int,
        qudit: int,
    ) -> SimpleCircuitPoint:
        return super().__new__(cls, (cycle, qudit))

    @property
    def cycle(self) -> int:
        """The point's cycle index."""
        return self[0]

    @property
    def qudit(self) -> int:
        """The point's qudit index."""
        return self[1]


class SingleQuditIterator(Iterator[Tuple[int, Operation]]):
    """A circuit iterator designed for walking down a single qudit."""

    def __init__(
        self,
        circuit: Circuit,
        qudit: int,
        start: int,
        end: int | None = None,
    ) -> None:
        """Construct a SingleQuditIterator."""
        self.qudit = qudit
        self.circuit = circuit
        self.start = start
        self.end = end if end is not None and \
            end < circuit.num_cycles else circuit.num_cycles
        self.cycle = start

    def __iter__(self) -> Iterator[tuple[int, Operation]]:
        self.cycle = self.start
        return self

    def __next__(self) -> tuple[int, Operation]:
        """Get the next operation."""
        if self.cycle >= self.end:
            raise StopIteration
        op = self.circuit._circuit[self.cycle][self.qudit]
        while op is None:
            self.cycle += 1
            if self.cycle >= self.end:
                raise StopIteration
            op = self.circuit._circuit[self.cycle][self.qudit]

        cycle_to_return = self.cycle
        self.cycle += 1
        return cycle_to_return, op


class CachedSingleQuditIterator(Iterator[Tuple[int, Operation]]):
    """A SingleQuditIterator which walks down a cache rather than directly
    walking the qudit."""

    @staticmethod
    def make_gate_cache(
        circuit: Circuit,
        qudit: int,
        multiqudit_only: bool = False,
    ) -> list[int]:
        """Creates the gate cache for a given circuit `circuit`, composed of
        gates with more than one input."""

        circuit_iterator = SingleQuditIterator(circuit, qudit, 0)
        gate_cache: list[int] = [
            cycle for cycle,
            op in circuit_iterator
            if len(op.location) > 1 or not multiqudit_only
        ]

        return gate_cache

    def __init__(
        self,
        circuit: Circuit,
        cache: Sequence[int],
        qudit: int,
        start: int,
        end: int | None = None,
    ) -> None:
        """Construct a CachedSingleQuditIterator."""
        self.circuit = circuit
        self.cache = cache
        self.qudit = qudit
        self.start_index = bisect.bisect_left(self.cache, start)
        self.end = end if end is not None and \
            end < circuit.num_cycles else circuit.num_cycles
        self.cache_index = self.start_index

    def __iter__(self) -> Iterator[tuple[int, Operation]]:
        # get an iterator to the cache, starting at the starting_point
        self.cache_index = self.start_index
        return self

    def __next__(self) -> tuple[int, Operation]:
        """Get the next operation."""
        if self.cache_index >= len(self.cache):
            raise StopIteration
        cycle_to_return = self.cache[self.cache_index]
        if cycle_to_return >= self.end:
            raise StopIteration

        op = self.circuit._circuit[cycle_to_return][self.qudit]
        if op is None:
            raise ValueError(
                f'Error, qudit cache contains None at {cycle_to_return}',
            )
        self.cache_index += 1
        return cycle_to_return, op
