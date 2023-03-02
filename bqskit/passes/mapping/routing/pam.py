"""This module implements the PAMRoutingPass."""
from __future__ import annotations

import copy
import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.point import CircuitPoint
from bqskit.passes.control.foreach import ForEachBlockPass

from bqskit.passes.mapping.pam import PermutationAwareMappingAlgorithm

_logger = logging.getLogger(__name__)


class PAMRoutingPass(PermutationAwareMappingAlgorithm, BasePass):

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        model = self.get_model(circuit, data)
        placement = self.get_placement(circuit, data)
        subgraph = model.coupling_graph.get_subgraph(placement)

        perm_data: dict[CircuitPoint, dict[tuple[int, ...], Circuit]] = {}
        block_datas = data[ForEachBlockPass.key][-1]
        for block_data in block_datas:
            perm_data[block_data['point']] = block_data['permutation_data']

        if not subgraph.is_fully_connected():
            raise RuntimeError('Cannot route circuit on disconnected qudits.')

        if 'initial_mapping' in data:
            pi = copy.deepcopy(data['initial_mapping'])
        else:
            pi = [i for i in range(circuit.num_qudits)]

        self.forward_pass(circuit, pi, subgraph, perm_data, modify_circuit=True)
        if 'final_mapping' in data:
            self._apply_perm(data['final_mapping'], pi)
        data['final_mapping'] = pi
        _logger.info(f'Finished routing with layout: {str(pi)}')
