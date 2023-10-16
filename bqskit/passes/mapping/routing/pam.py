"""This module implements the PAMRoutingPass."""
from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.point import CircuitPoint
from bqskit.passes.control.foreach import ForEachBlockPass
from bqskit.passes.mapping.pam import PAMBlockTAPermData
from bqskit.passes.mapping.pam import PermutationAwareMappingAlgorithm

_logger = logging.getLogger(__name__)


class PAMRoutingPass(PermutationAwareMappingAlgorithm, BasePass):

    out_data_key = '_pam_routing_block_out_data'

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        subgraph = data.connectivity
        if not subgraph.is_fully_connected():
            raise RuntimeError('Cannot route circuit on disconnected qudits.')

        perm_data: dict[CircuitPoint, PAMBlockTAPermData] = {}
        block_datas = data[ForEachBlockPass.key][-1]
        for block_data in block_datas:
            perm_data[block_data['point']] = block_data['permutation_data']

        pi = [i for i in range(circuit.num_qudits)]
        out_data = self.forward_pass(circuit, pi, subgraph, perm_data, True)
        data.final_mapping = [pi[x] for x in data.final_mapping]

        _logger.info(f'Finished routing with layout: {str(pi)}')
        data[self.out_data_key] = out_data
