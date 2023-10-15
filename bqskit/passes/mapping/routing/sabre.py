"""This module implements the GeneralizedSabreRoutingPass class."""
from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.passes.mapping.sabre import GeneralizedSabreAlgorithm


_logger = logging.getLogger(__name__)


class GeneralizedSabreRoutingPass(BasePass, GeneralizedSabreAlgorithm):
    """
    Uses the Sabre algorithm to route the circuit.

    See :class:`GeneralizedSabreAlgorithm` for more info.
    """

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        subgraph = data.connectivity
        if not subgraph.is_fully_connected():
            raise RuntimeError('Cannot route circuit on disconnected qudits.')

        pi = [i for i in range(circuit.num_qudits)]
        self.forward_pass(circuit, pi, subgraph, modify_circuit=True)
        data.final_mapping = [pi[x] for x in data.initial_mapping]

        _logger.info(f'Finished routing with layout: {str(pi)}')
