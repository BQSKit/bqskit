"""This module implements the TrivialPlacementPass class."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
_logger = logging.getLogger(__name__)


class TrivialPlacementPass(BasePass):
    """Place the logical qubits on the first n physical qubits."""

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""

        # Select the first n physical qubits for placement
        trivial_placement = list(range(circuit.num_qudits))
        model = BasePass.get_model(circuit, data)
        data['physical_qudits'] = trivial_placement

        _logger.info(f'Placed qudits on {data["physical_qudits"]}')

        # Raise an error if this is not a valid placement
        sg = model.coupling_graph.get_subgraph(data['physical_qudits'])
        if not sg.is_fully_connected():
            raise RuntimeError('The trivial placement is not valid.')
