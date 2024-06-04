"""This module implements the StaticPlacementPass class."""

from __future__ import annotations

import logging

from timeout_decorator import timeout, TimeoutError
import networkx as nx

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
        timeout_sec: float,
    ) -> list[int]:
        """Find an monomorphic subgraph."""

        @timeout(timeout_sec)
        def _find_monomorphic_subgraph(
            physical_graph: CouplingGraph, logical_graph: CouplingGraph
        ) -> list[int]:

            # Convert the coupling graph to a networkx graph
            def coupling_graph_to_nx_graph(coupling_graph: CouplingGraph) -> nx.Graph:
                nx_graph = nx.Graph()
                nx_graph.add_nodes_from(range(coupling_graph.num_qudits))
                nx_graph.add_edges_from([e for e in coupling_graph])
                return nx_graph

            # Find an monomorphic subgraph
            graph_matcher = nx.algorithms.isomorphism.GraphMatcher(
                coupling_graph_to_nx_graph(physical_graph),
                coupling_graph_to_nx_graph(logical_graph),
            )
            if not graph_matcher.subgraph_is_monomorphic():
                return []

            placement = list(range(logical_graph.num_qudits))
            for k, v in graph_matcher.mapping.items():
                placement[v] = k
            return placement

        return _find_monomorphic_subgraph(physical_graph, logical_graph)

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        physical_graph = data.model.coupling_graph
        logical_graph = circuit.coupling_graph

        # Find an monomorphic subgraph
        try:
            data.placement = self.find_monomorphic_subgraph(
                physical_graph, logical_graph, self.timeout_sec
            )
            if len(data.placement) == 0:
                raise RuntimeError("No monomorphic subgraph found.")
        except TimeoutError:
            raise RuntimeError("Static placement search timed out.")

        _logger.info(f"Placed qudits on {data.placement}")

        # Raise an error if this is not a valid placement
        if not all(
            data.placement[e[1]]
            in physical_graph.get_neighbors_of(data.placement[e[0]])
            for e in logical_graph
        ):
            raise RuntimeError("No valid placement found.")
