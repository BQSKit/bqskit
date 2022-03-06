"""This module implements the MachineModel class."""
from __future__ import annotations

import itertools as it
from functools import lru_cache
from typing import Iterable

from bqskit.ir.location import CircuitLocation
from bqskit.ir.location import CircuitLocationLike
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_valid_coupling_graph


class MachineModel:
    """A model of a quantum processing unit's connectivity."""

    def __init__(
        self,
        num_qudits: int,
        coupling_graph: Iterable[tuple[int, int]] | None = None,
    ) -> None:
        """
        MachineModel Constructor.

        Args:
            num_qudits (int): The total number of qudits in the machine.

            coupling_graph (Iterable[tuple[int, int]] | None): A coupling
                graph describing which pairs of qudits can interact.
                Given as an undirected edge set. If left as None, then
                an all-to-all coupling graph is used as a default.
                (Default: None)

        Raises:
            ValueError: If `num_qudits` is nonpositive.
        """

        if not is_integer(num_qudits):
            raise TypeError(
                f'Expected integer num_qudits, got {type(num_qudits)}.',
            )

        if num_qudits <= 0:
            raise ValueError(f'Expected positive num_qudits, got {num_qudits}.')

        if coupling_graph is None:
            coupling_graph = set(it.combinations(range(num_qudits), 2))

        if not is_valid_coupling_graph(coupling_graph, num_qudits):
            raise TypeError('Invalid coupling graph.')

        self.coupling_graph = set(coupling_graph)
        self.num_qudits = num_qudits

        self._adjacency_list: list[list[int]] = [[] for _ in range(num_qudits)]
        for q0, q1 in self.coupling_graph:
            self._adjacency_list[q0].append(q1)
            self._adjacency_list[q1].append(q0)

    @lru_cache(maxsize=None)
    def get_locations(self, block_size: int) -> list[CircuitLocation]:
        """
        Returns a list of locations that complies with the machine.

        Each location describes a valid spot for a `block_size`-sized gate,
        so the number of qudit_indices in each location is `block_size`.
        A location is only included if each pair of qudits is directly
        connected or connected through other qudits in the location.

        Args:
            block_size (int): The size of each location in the final list.

        Returns:
            list[CircuitLocation]: The locations compliant with the machine.

        Raises:
            ValueError: If `block_size` is nonpositive or too large.

        Notes:
            Does a breadth first search on all pairs of qudits, keeps paths
            that have length equal to block_size. Note that the coupling map
            is assumed to be undirected.
        """

        if not is_integer(block_size):
            raise TypeError(
                f'Expected integer for block_size, got {type(block_size)}',
            )

        if block_size > self.num_qudits:
            raise ValueError(
                'The block_size is too large; '
                f'expected <= {self.num_qudits}, got {block_size}.',
            )

        if block_size <= 0:
            raise ValueError(f'Nonpositive block_size; got {block_size}.')

        locations: set[CircuitLocation] = set()

        for qudit in range(self.num_qudits):
            # Get every valid set containing `qudit` with size == block_size
            self._location_search(locations, set(), qudit, block_size)

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
            for qudit in self._adjacency_list[node]
            if qudit not in curr_path
        }

        for neighbor in frontier:
            self._location_search(locations, curr_path, neighbor, limit)

    def get_subgraph(
        self,
        location: CircuitLocationLike,
        renumbering: dict[int, int] | None = None,
    ) -> list[tuple[int, int]]:
        """Returns the sub_coupling_graph with qudits in `location`."""
        if not CircuitLocation.is_location(location, self.num_qudits):
            raise TypeError('Invalid location.')

        location = CircuitLocation(location)
        if renumbering is None:
            renumbering = {x: x for x in range(self.num_qudits)}

        subgraph = []
        for q0, q1 in self.coupling_graph:
            if q0 in location and q1 in location:
                subgraph.append((renumbering[q0], renumbering[q1]))
        return subgraph
