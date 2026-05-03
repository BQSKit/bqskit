from __future__ import annotations
# Generic utils

import heapq
from typing import Set, Iterable, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison
    
T = TypeVar('T', bound='SupportsRichComparison')
class PriorityQueueSet(Set[T], object):
    """
    Source: https://stackoverflow.com/questions/407734/a-generic-priority-queue-for-python
    Combined priority queue and set data structure.

    Acts like a priority queue, except that its items are guaranteed to be
    unique. Provides O(1) membership test, O(log N) insertion and O(log N)
    removal of the smallest item.

    Important: the items of this data structure must be both comparable and
    hashable (i.e. must implement __cmp__ and __hash__). This is true of
    Python's built-in objects, but you should implement those methods if you
    want to use the data structure for custom objects.
    """
    
    def __init__(self, items : Iterable[T] = tuple()) -> None:
        """
        Create a new PriorityQueueSet.

        Arguments:
            items (list): An initial item list - it can be unsorted and
                non-unique. The data structure will be created in O(N).
        """
        self.set : set[T] = set(items)
        self.heap : list[T] = list(items)
        heapq.heapify(self.heap)

    def __contains__(self, item : object) -> bool:
        return item in self.set
    
    def __len__(self) -> int:
        return len(self.set)

    def pop(self) -> T:
        """Remove and return the smallest item from the queue."""
        smallest = heapq.heappop(self.heap)
        self.set.remove(smallest)
        return smallest

    def push(self, item : T) -> None:
        """Add ``item`` to the queue if doesn't already exist."""
        if item not in self.set:
            self.set.add(item)
            heapq.heappush(self.heap, item)
                
# BQSKit utils
from typing import Iterator
from typing import Tuple
from typing import Sequence
import bisect

from bqskit.ir.operation import Operation
from bqskit.ir.circuit import Circuit

class SimpleCircuitPoint(Tuple[int, int]):
    """
    A simple, more efficient version of BQSKit's CircuitPoint.
    """
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
        starting_point: SimpleCircuitPoint,
        circuit: Circuit,
    ) -> None:
        """Construct a SingleQuditIterator."""
        self.qudit = starting_point.qudit
        self.circuit = circuit
        self.num_cycles = circuit.num_cycles
        self.cycle = starting_point.cycle

    def __iter__(self) -> Iterator[Tuple[int, Operation]]:
        return self

    def __next__(self) -> Tuple[int, Operation]:
        """Get the next operation."""
        if self.cycle >= self.num_cycles:
            raise StopIteration
        op = self.circuit._circuit[self.cycle][self.qudit]
        while op is None:
            self.cycle += 1
            if self.cycle >= self.num_cycles:
                raise StopIteration
            op = self.circuit._circuit[self.cycle][self.qudit]
            
        cycle_to_return = self.cycle
        self.cycle += 1
        return cycle_to_return, op
    
class CachedSingleQuditIterator(Iterator[Tuple[int, Operation]]):
    """A SingleQuditIterator which walks down a cache rather than directly walking the qudit."""
    
    @staticmethod
    def make_multiqudit_gate_cache(circuit: Circuit, qudit: int) -> list[int]:
        """
        Creates the gate cache for a given circuit `circuit`, composed of gates with more than one input.
        """
        
        starting_point = SimpleCircuitPoint(0, qudit)
        circuit_iterator = SingleQuditIterator(starting_point, circuit)
        gate_cache: list[int] = [cycle for cycle, op in circuit_iterator if len(op.location._location) > 1]
            
        return gate_cache

    def __init__(
        self,
        starting_point: SimpleCircuitPoint,
        circuit: Circuit,
        cache: Sequence[int]
    ) -> None:
        """Construct a CachedSingleQuditIterator."""
        self.qudit = starting_point.qudit
        self.circuit = circuit
        # get an iterator to the cache, starting at the starting_point
        self.cache_iter = iter(cache[i] for i in range(bisect.bisect_left(cache, starting_point.cycle), len(cache)))

    def __iter__(self) -> Iterator[Tuple[int, Operation]]:
        return self

    def __next__(self) -> Tuple[int, Operation]:
        """Get the next operation."""
        cycle_to_return = next(self.cache_iter)
        op = self.circuit._circuit[cycle_to_return][self.qudit]
        if op is None:
            raise ValueError(f'Error, qudit cache contains None at {cycle_to_return}')
        return cycle_to_return, op