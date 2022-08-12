"""This module implements the GeneralizedSabreLayoutPass class."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.passes.mapping.sabre import GeneralizedSabreAlgorithm

_logger = logging.getLogger(__name__)


class GeneralizedSabreLayoutPass(BasePass, GeneralizedSabreAlgorithm):
    """
    Uses the Sabre algorithm to layout the circuit.

    See :class:`GeneralizedSabreAlgorithm` for more info.
    """

    def __init__(
        self,
        total_passes: int = 1,
        decay_delta: float = 0.001,
        decay_reset_interval: int = 5,
        decay_reset_on_gate: bool = True,
        extended_set_size: int = 20,
        extended_set_weight: float = 0.5,
    ) -> None:
        """
        Construct a GeneralizedSabreLayoutPass.

        Args:
            total_passes (int): The amount of forward and backward passes
                to apply before finalizing the layout.

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
            raise TypeError(
                'Expected int for total_passes'
                f', got {type(total_passes)}',
            )

        if total_passes < 1:
            raise ValueError('Total passes must be a positive integer.')

        self.total_passes = total_passes
        super().__init__(
            decay_delta,
            decay_reset_interval,
            decay_reset_on_gate,
            extended_set_size,
            extended_set_weight,
        )

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        model = self.get_model(circuit, data)
        placement = self.get_placement(circuit, data)
        subgraph = model.coupling_graph.get_subgraph(placement)

        if not subgraph.is_fully_connected():
            raise RuntimeError('Cannot layout circuit on disconnected qudits.')

        pi = [i for i in range(circuit.num_qudits)]
        for _ in range(self.total_passes):
            self.forward_pass(circuit, pi, subgraph)
            self.backward_pass(circuit, pi, subgraph)

        # select qubits
        _logger.info(f'Found layout: {str(pi)}')
        data['initial_mapping'] = pi
