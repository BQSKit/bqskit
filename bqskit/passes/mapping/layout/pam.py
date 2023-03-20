"""This module implements the PAMLayoutPass."""
from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.point import CircuitPoint
from bqskit.passes.control.foreach import ForEachBlockPass
from bqskit.passes.mapping.pam import PermutationAwareMappingAlgorithm

_logger = logging.getLogger(__name__)


class PAMLayoutPass(PermutationAwareMappingAlgorithm, BasePass):
    """Layout algorithm using permutation-aware mapping."""

    def __init__(
        self,
        total_passes: int = 1,
        gate_count_weight: float = 0.3,
        decay_delta: float = 0.001,
        decay_reset_interval: int = 5,
        decay_reset_on_gate: bool = True,
        extended_set_size: int = 20,
        extended_set_weight: float = 0.5,
    ) -> None:
        """
        Construct a PAMLayoutPass.

        Args:
            total_passes (int): The amount of forward and backward passes
                to apply before finalizing the layout.

            gate_count_weight (float): See
                :class:`PermutationAwareMappingAlgorithm` for info.
                (Default: 0.3)

            decay_delta (float): See :class:`GeneralizedSabreAlgorithm`
                for info. (Default: 0.001)

            decay_reset_interval (int): See :class:`GeneralizedSabreAlgorithm`
                for info. (Default: 5)

            decay_reset_on_gate (bool): See :class:`GeneralizedSabreAlgorithm`
                for info. (Default: True)

            extended_set_size (int): See :class:`GeneralizedSabreAlgorithm`
                for info. (Default: 20)

            extended_set_weight (float): See :class:`GeneralizedSabreAlgorithm`
                for info. (Default: 0.5)
        """
        if not isinstance(total_passes, int):
            m = f'Expected int for total_passes, got {type(total_passes)}'
            raise TypeError(m)

        if total_passes < 1:
            raise ValueError('Total passes must be a positive integer.')

        self.total_passes = total_passes
        super().__init__(
            gate_count_weight,
            decay_delta,
            decay_reset_interval,
            decay_reset_on_gate,
            extended_set_size,
            extended_set_weight,
        )

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        subgraph = data.connectivity

        perm_data: dict[CircuitPoint, dict[tuple[int, ...], Circuit]] = {}
        block_datas = data[ForEachBlockPass.key][-1]
        for block_data in block_datas:
            perm_data[block_data['point']] = block_data['permutation_data']

        if not subgraph.is_fully_connected():
            raise RuntimeError('Cannot route circuit on disconnected qudits.')

        pi = [i for i in range(circuit.num_qudits)]
        for _ in range(self.total_passes):
            self.forward_pass(circuit, pi, subgraph, perm_data)
            self.backward_pass(circuit, pi, subgraph)

        data['initial_mapping'] = pi
        _logger.info(f'Found layout: {str(pi)}')
