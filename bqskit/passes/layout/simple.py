"""This module implements the SimpleLayoutPass class."""
from __future__ import annotations

from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.machine import MachineModel
from bqskit.ir.circuit import Circuit


class SimpleLayoutPass(BasePass):
    """
    Assigns logical qudits to physical qudits.

    Layout also encodes a machine model into the compilation workflow.
    """

    def __init__(self, model: MachineModel | None = None) -> None:
        """
        Construct a SimpleLayoutPass.

        Args:
            model (MachineModel | None): The machine model to use for
                layout and to encode into the compilation workflow.
                Defaults to an all-to-all model matching the size of the
                circuit.
        """
        if model is not None and not isinstance(model, MachineModel):
            raise TypeError(f'Expected MachineModel, got: {type(model)}.')

        self.model = model

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        if self.model is None:
            data['machine_model'] = MachineModel(circuit.num_qudits)
            return

        if self.model.num_qudits < circuit.num_qudits:
            raise RuntimeError('Machine model is too small for circuit.')

        graph = self.model.get_subgraph(list(range(circuit.num_qudits)))
        data['machine_model'] = MachineModel(circuit.num_qudits, graph)
