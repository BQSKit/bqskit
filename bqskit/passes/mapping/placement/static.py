"""This module implements the StaticPlacementPass class."""

from __future__ import annotations

import logging
import itertools
import time

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.qis.graph import CouplingGraph

_logger = logging.getLogger(__name__)


class StaticPlacementPass(BasePass):
    """Find a subgraph monomorphic to the coupling graph so that no SWAPs are needed."""

    def __init__(self, timeout_sec: float = 10) -> None:
        self.timeout_sec = timeout_sec

    def find_monomorphic_subgraph(
        self,
        physical_graph: CouplingGraph,
        logical_graph: CouplingGraph,
    ) -> list[int]:
        """Try all possible placements"""

        start_time = time.time()
        for placement in itertools.permutations(
            range(physical_graph.num_qudits), logical_graph.num_qudits
        ):
            if time.time() - start_time > self.timeout_sec:
                return []

            placement = list(placement)
            if all(
                placement[e[1]] in physical_graph.get_neighbors_of(placement[e[0]])
                for e in logical_graph
            ):
                return placement

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        physical_graph = data.model.coupling_graph
        logical_graph = circuit.coupling_graph

        # Find an monomorphic subgraph
        placement = self.find_monomorphic_subgraph(physical_graph, logical_graph)

        # Set the placement if it is valid
        if len(placement) == logical_graph.num_qudits and all(
            placement[e[1]] in physical_graph.get_neighbors_of(placement[e[0]])
            for e in logical_graph
        ):
            data.placement = placement
            _logger.info(f"Placed qudits on {data.placement}")
