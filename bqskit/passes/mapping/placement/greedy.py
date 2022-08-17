"""This module implements the GreedyPlacementPass class."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
_logger = logging.getLogger(__name__)


class GreedyPlacementPass(BasePass):
    """Find a placement by starting with the most connected qudit."""

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""

        # Find physical qudit with highest degree
        graph = self.get_model(circuit, data).coupling_graph
        degrees = np.array(graph.get_qudit_degrees())
        highest_degree_qudit = int(np.argmax(degrees))

        # Search for placement by greedily adding best neighbor
        placement = [highest_degree_qudit]
        neighbors = graph.get_neighbors_of(highest_degree_qudit)
        while len(placement) < circuit.num_qudits:
            # Score each neighbor of current placement set
            best_score = None
            best_neighbor = None
            for q in neighbors:
                inter = [n for n in graph.get_neighbors_of(q) if n in placement]
                score = (len(inter), degrees[q])
                if best_score is None or score > best_score:
                    best_score = score
                    best_neighbor = q

            # Add best scoring neighbor to placement
            assert best_neighbor is not None
            neighbors.remove(best_neighbor)
            placement.append(best_neighbor)
            for n in graph.get_neighbors_of(best_neighbor):
                if n not in placement and n not in neighbors:
                    neighbors.append(n)

        data['physical_qudits'] = sorted(placement)

        _logger.info(f'Placed qudits on {data["physical_qudits"]}')

        # Raise an error if this is not a valid placement
        sg = graph.get_subgraph(data['physical_qudits'])
        if not sg.is_fully_connected():
            raise RuntimeError('No valid placement found.')
