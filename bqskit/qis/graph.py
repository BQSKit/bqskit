"""This module implements the CouplingGraph class."""
from __future__ import annotations

import itertools as it
import logging
from functools import lru_cache
from random import shuffle
from typing import Any
from typing import Collection
from typing import Iterable
from typing import Iterator
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

if TYPE_CHECKING:
    from typing_extensions import TypeGuard

from bqskit.ir.location import CircuitLocation
from bqskit.ir.location import CircuitLocationLike
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_iterable

_logger = logging.getLogger(__name__)


class CouplingGraph(Collection[Tuple[int, int]]):
    """A graph representing connections in a qudit topology."""

    def __init__(self, graph: Iterable[tuple[int, int]]) -> None:
        if isinstance(graph, CouplingGraph):
            self.num_qudits: int = graph.num_qudits
            self._edges: set[tuple[int, int]] = graph._edges
            self._adj: list[list[int]] = graph._adj
            return

        if not CouplingGraph.is_valid_coupling_graph(graph):
            raise TypeError('Invalid coupling graph.')

        self._edges = set(graph)

        self.num_qudits = 0
        for q1, q2 in self._edges:
            self.num_qudits = max(self.num_qudits, max(q1, q2))
        self.num_qudits += 1

        self._adj = [[] for _ in range(self.num_qudits)]
        for q1, q2 in self._edges:
            self._adj[q1].append(q2)
            self._adj[q2].append(q1)

    def is_fully_connected(self) -> bool:
        """Return true if the graph is fully connected."""
        frontier = {0}
        qudits_seen = {0}

        while len(frontier) > 0:
            expanded_qudits = set()

            for qudit in frontier:
                expanded_qudits.update(self._adj[qudit])

            frontier = expanded_qudits - qudits_seen
            qudits_seen.update(expanded_qudits)

            if len(qudits_seen) == self.num_qudits:
                return True

        return False

    def get_neighbors_of(self, qudit: int) -> list[int]:
        """Return the qudits adjacent to `qudit`."""
        return self._adj[qudit]

    def __contains__(self, __o: object) -> bool:
        return self._edges.__contains__(__o)

    def __eq__(self, __o: object) -> bool:
        return self._edges.__eq__(__o)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        return self._edges.__iter__()

    def __hash__(self) -> int:
        return hash(tuple(self._edges))

    def __len__(self) -> int:
        return self._edges.__len__()

    def __str__(self) -> str:
        return 'CouplingGraph(' + self._edges.__str__() + ')'

    def __repr__(self) -> str:
        return self._edges.__repr__()

    def get_qudit_degrees(self) -> list[int]:
        return [len(l) for l in self._adj]

    def get_subgraph(
        self,
        location: CircuitLocationLike,
        renumbering: dict[int, int] | None = None,
    ) -> CouplingGraph:
        """Returns the sub-coupling-graph with qudits in `location`."""
        if not CircuitLocation.is_location(location, self.num_qudits):
            raise TypeError('Invalid location.')

        location = CircuitLocation(location)
        if renumbering is None:
            renumbering = {x: x for x in range(self.num_qudits)}

        subgraph = []
        for q0, q1 in self._edges:
            if q0 in location and q1 in location:
                subgraph.append((renumbering[q0], renumbering[q1]))
        return CouplingGraph(subgraph)

    @lru_cache(maxsize=None)
    def get_subgraphs_of_size(self, size: int) -> list[CircuitLocation]:
        """
        Find all sets of indices that form connected subgraphs on their own.

        Each location describes a valid spot for a `size`-sized gate,
        so the number of qudit_indices in each location is `size`.
        A location is only included if each pair of qudits is directly
        connected or connected through other qudits in the location.

        Args:
            size (int): The size of each location in the final list.

        Returns:
            list[CircuitLocation]: The locations compliant with the machine.

        Raises:
            ValueError: If `size` is nonpositive or too large.

        Notes:
            Does a breadth first search on all pairs of qudits, keeps paths
            that have length equal to size. Note that the coupling map
            is assumed to be undirected.
        """
        if not is_integer(size):
            raise TypeError(
                f'Expected integer for block_size, got {type(size)}',
            )

        if size > self.num_qudits:
            raise ValueError(
                'The block_size is too large; '
                f'expected <= {self.num_qudits}, got {size}.',
            )

        if size <= 0:
            raise ValueError(f'Nonpositive size; got {size}.')

        locations: set[CircuitLocation] = set()

        for qudit in range(self.num_qudits):
            # Get every valid set containing `qudit` with size == size
            self._location_search(locations, set(), qudit, size)

        return list(locations)

    def _location_search(
        self,
        locations: set[CircuitLocation],
        path: set[int],
        vertex: int,
        limit: int,
    ) -> None:
        """
        Add paths with length equal to limit to the `locations` set.

        Args:
            locations (set[CircuitLocation]): A list that contains all paths
                found so far of length equal to `limit`.

            path (set[int]): The qudit vertices currently included in
                the path.

            vertex (int): The vertex in the graph currently being examined.

            limit (int): The desired length of paths in the `locations`
                list.
        """
        if vertex in path:
            return

        curr_path = path.copy()
        curr_path.add(vertex)

        if len(curr_path) == limit:
            locations.add(CircuitLocation(list(curr_path)))
            return

        frontier: set[int] = {
            qudit
            for node in curr_path
            for qudit in self._adj[node]
            if qudit not in curr_path
        }

        for neighbor in frontier:
            self._location_search(locations, curr_path, neighbor, limit)

    @staticmethod
    def is_valid_coupling_graph(
        coupling_graph: Any,
        num_qudits: int | None = None,
    ) -> TypeGuard[CouplingGraphLike]:
        """
        Return true if the coupling graph is valid.

        Args:
            coupling_graph (Any): The coupling graph to check.

            num_qudits (int | None): The total number of qudits. All qudits
                should be less than this. If None, don't check.

        Returns:
            bool: Valid or not
        """

        if not is_iterable(coupling_graph):
            _logger.debug('Coupling graph is not iterable.')
            return False

        if len(list(coupling_graph)) == 0:
            return True

        if not all(isinstance(pair, tuple) for pair in coupling_graph):
            _logger.debug('Coupling graph is not a sequence of tuples.')
            return False

        if not all([len(pair) == 2 for pair in coupling_graph]):
            _logger.debug('Coupling graph is not a sequence of pairs.')
            return False

        if num_qudits is not None:
            if not (is_integer(num_qudits) and num_qudits > 0):
                _logger.debug('Invalid num_qudits in coupling graph check.')
                return False

            if not all(
                qudit < num_qudits
                for pair in coupling_graph
                for qudit in pair
            ):
                _logger.debug('Coupling graph has invalid qudits.')
                return False

        if not all([
            len(pair) == len(set(pair))
            for pair in coupling_graph
        ]):
            _logger.debug('Coupling graph has an invalid pair.')
            return False

        return True

    @staticmethod
    def all_to_all(num_qudits: int) -> CouplingGraph:
        return CouplingGraph(set(it.combinations(range(num_qudits), 2)))

    @staticmethod
    def linear(num_qudits: int) -> CouplingGraph:
        return CouplingGraph([(x, x + 1) for x in range(num_qudits - 1)])

    def maximal_matching(
        self,
        edges_to_ignore: list[tuple[int, int]] = [],
        randomize: bool = False,
    ) -> list[tuple[int, int]]:
        """
        Generate a random graph matching for the coupling graph. Edges in the
        `ignored_edges` list will not be included.

        Arguments:
            edges_to_ignore (list[tuple[int]]): Edges not included in the
                matching. (Default: [])

            randomize (bool): Shuffle edges to create random matchings.
                (Default: False)

        Returns:
            matching (list[tuple[int]]): A maximal list of edges that share
                no common verticies.
        """
        matching: set[tuple[int, int]] = set()
        vertices: set[int] = set()

        edge_list = [
            (u, v) for (u, v) in self._edges if (u, v) not in edges_to_ignore
            and (v, u) not in edges_to_ignore
        ]

        if randomize:
            shuffle(edge_list)

        for edge in edge_list:
            u, v = edge
            if u not in vertices and v not in vertices and u != v:
                matching.add(edge)
                vertices.update(edge)
        return list(matching)


CouplingGraphLike = Union[Iterable[Tuple[int, int]], CouplingGraph]
