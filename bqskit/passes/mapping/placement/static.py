"""This module implements the StaticPlacementPass class."""
from __future__ import annotations

import logging
import time

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.qis.graph import CouplingGraph

_logger = logging.getLogger(__name__)


class StaticPlacementPass(BasePass):
    """Find a subgraph monomorphic to the coupling graph so that no SWAPs are
    needed."""

    def __init__(self, timeout_sec: float = 10) -> None:
        self.timeout_sec = timeout_sec

    def _find_monomorphic_subgraph(
        self,
        time_limit: float,
        physical_graph: CouplingGraph,
        num_logical_qudits: int,
        minimal_degrees: list[int],
        connected_indices: list[list[int]],
        current_placement: list[int] = [],
        current_index: int = 0,
    ) -> list[int]:
        """Recursively find a monomorphic subgraph."""
        if current_index == num_logical_qudits:
            return current_placement

        if time.time() > time_limit:
            return []

        # Find all possible placements for the current logical qudit
        candidate_indices = set()

        # Filter out occupied qudits and qudits with insufficient degrees
        physical_degrees = physical_graph.get_qudit_degrees()
        for x in range(physical_graph.num_qudits):
            if (
                physical_degrees[x] >= minimal_degrees[current_index]
                and x not in current_placement
            ):
                candidate_indices.add(x)

        # Filter out qudits that are not connected to previous logical qudits
        for i in connected_indices[current_index]:
            candidate_indices &= set(
                physical_graph.get_neighbors_of(current_placement[i]),
            )

        # Try all possible placements for the current logical qudit
        for x in candidate_indices:
            new_placement = current_placement + [x]
            result = self._find_monomorphic_subgraph(
                time_limit,
                physical_graph,
                num_logical_qudits,
                minimal_degrees,
                connected_indices,
                new_placement,
                current_index + 1,
            )
            if len(result) == num_logical_qudits:
                return result

        # If no valid placement is found, return an empty list
        return []

    def find_monomorphic_subgraph(
        self,
        physical_graph: CouplingGraph,
        logical_graph: CouplingGraph,
    ) -> list[int]:
        """Try all possible placements."""

        # To be optimized later
        logical_qubit_order = list(range(logical_graph.num_qudits))

        minimum_degrees = [
            logical_graph.get_qudit_degrees()[i] for i in logical_qubit_order
        ]
        connected_indices: list[list[int]] = [
            [] for _ in range(logical_graph.num_qudits)
        ]
        for i in range(logical_graph.num_qudits):
            for j in range(i):
                if logical_qubit_order[j] in logical_graph.get_neighbors_of(
                    logical_qubit_order[i],
                ):
                    connected_indices[i].append(j)

        # Find a monomorphic subgraph
        start_time = time.time()
        index_to_physical = self._find_monomorphic_subgraph(
            start_time + self.timeout_sec,
            physical_graph,
            logical_graph.num_qudits,
            minimum_degrees,
            connected_indices,
        )
        _logger.info(f'elapsed time: {time.time() - start_time}')
        if len(index_to_physical) == 0:
            return []

        # Convert the result to a placement
        placement = [-1] * logical_graph.num_qudits
        for i, x in enumerate(logical_qubit_order):
            placement[x] = index_to_physical[i]
        return placement

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        physical_graph = data.model.coupling_graph
        logical_graph = circuit.coupling_graph

        # Find an monomorphic subgraph
        placement = self.find_monomorphic_subgraph(
            physical_graph, logical_graph,
        )

        # Set the placement if it is valid
        if len(placement) == logical_graph.num_qudits and all(
            placement[e[1]] in physical_graph.get_neighbors_of(placement[e[0]])
            for e in logical_graph
        ):
            data.placement = placement
            _logger.info(f'Placed qudits on {data.placement}')
        else:
            _logger.info('No valid placement found')
