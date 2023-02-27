"""This module implements the ApplyPlacement class."""
from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit

_logger = logging.getLogger(__name__)


class ApplyPlacement(BasePass):
    """Place the circuit on the machine model."""

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        model = self.get_model(circuit, data)
        placement = self.get_placement(circuit, data)
        physical_circuit = Circuit(model.num_qudits, model.radixes)
        physical_circuit.append_circuit(circuit, placement)
        circuit.become(physical_circuit)
        if 'final_mapping' in data:
            pi = data['final_mapping']
            data['final_mapping'] = [placement[p] for p in pi]
        data['placement'] = list(i for i in range(model.num_qudits))
