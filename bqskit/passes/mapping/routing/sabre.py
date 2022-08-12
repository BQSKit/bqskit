"""This module implements the GeneralizedSabreRoutingPass class."""
from __future__ import annotations

import copy
import logging
from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.passes.mapping.sabre import GeneralizedSabreAlgorithm


_logger = logging.getLogger(__name__)


class GeneralizedSabreRoutingPass(BasePass, GeneralizedSabreAlgorithm):
    """
    Uses the Sabre algorithm to route the circuit.

    See :class:`GeneralizedSabreAlgorithm` for more info.
    """

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        model = self.get_model(circuit, data)
        placement = self.get_placement(circuit, data)
        subgraph = model.coupling_graph.get_subgraph(placement)

        if not subgraph.is_fully_connected():
            raise RuntimeError('Cannot route circuit on disconnected qudits.')

        if 'initial_mapping' in data:
            pi = copy.deepcopy(data['initial_mapping'])
        else:
            pi = [i for i in range(circuit.num_qudits)]

        self.forward_pass(circuit, pi, subgraph, modify_circuit=True)
        data['final_mapping'] = pi
        _logger.info(f'Finished routing with layout: {str(pi)}')
