"""
This module implements the MachineModel class.

A MachineModel models a quantum processing unit's connectivity.
"""
from __future__ import annotations

import itertools as it
from typing import Optional
from typing import Sequence

from bqskit.utils.typing import is_valid_coupling_graph
from bqskit.utils.typing import is_valid_location


class MachineModel:
    """The MachineModel Class."""

    def __init__(
        self,
        num_qudits: int,
        coupling_graph: Optional[Sequence[tuple[int, int]]] = None,
    ) -> None:
        """
        MachineModel Constructor.

        If no coupling_graph is specified, this constructor produces a
        model with an all-to-all topology.

        Args:
            num_qudits (int): The total number of qudits in the machine.

            coupling_graph (Optional[Sequence[tuple[int, int]]]): List of
                connected qudit pairs. If None, then a fully-connected
                coupling_graph is used as a default.
        """

        if not isinstance(num_qudits, int):
            raise TypeError(
                'Expected integer num_qudits'
                ', got %s.' % type(num_qudits),
            )

        if num_qudits <= 0:
            raise ValueError(
                'Expected positive num_qudits'
                ', got %d.' % num_qudits,
            )

        if coupling_graph is None:
            coupling_graph = list(it.combinations(range(num_qudits), 2))

        if not is_valid_coupling_graph(coupling_graph, num_qudits):
            raise TypeError('Invalid coupling graph.')

        self.coupling_graph = list(coupling_graph)
        self.num_qudits = num_qudits
        self._cache: dict[int, list[tuple[int, ...]]] = {}

        self._adjacency_list: list[list[int]] = [[] for _ in range(num_qudits)]
        for q0, q1 in self.coupling_graph:
            self._adjacency_list[q0].append(q1)
            self._adjacency_list[q1].append(q0)

    def get_valid_locations(self, gate_size: int) -> list[tuple[int, ...]]:
        """
        Returns a list of locations that complies with the machine.

        Each location describes a valid spot for a `gate_size`-sized gate,
        so the number of qudit_indices in each location is `gate_size`.
        A location is only included if each pair of qubits is directly
        connected or connected through other qubits in the location.

        Args:
            gate_size (int): The size of each location in the final list.

        Returns:
            (list[tuple[int, ...]]): The locations compliant with the machine.

        Raises:
            ValueError: If the gate_size is nonpositive or too large.
        """

        if gate_size > self.num_qudits:
            raise ValueError('The gate_size is too large.')

        if gate_size <= 0:
            raise ValueError('The gate_size is nonpositive.')

        if gate_size in self._cache:
            return self._cache[gate_size]

        locations = []

        for group in it.combinations(range(self.num_qudits), gate_size):
            # Depth First Search
            seen = {group[0]}
            frontier = [group[0]]

            while len(frontier) > 0 and len(seen) < len(group):
                for q in group:
                    if frontier[0] in self._adjacency_list[q]:
                        if q not in seen:
                            seen.add(q)
                            frontier.append(q)

                frontier = frontier[1:]

            if len(seen) == len(group):
                locations.append(group)

        self._cache[gate_size] = locations
        return locations

    def get_subgraph(self, location: Sequence[int]) -> list[tuple[int, int]]:
        """Returns the sub_coupling_graph with qubits in location."""
        if not is_valid_location(location, self.num_qudits):
            raise TypeError('Invalid location.')

        subgraph = []
        for q0, q1 in self.coupling_graph:
            if q0 in location and q1 in location:
                subgraph.append((q0, q1))
        return subgraph
