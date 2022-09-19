"""This module implements the SetModelPass class."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.machine import MachineModel
from bqskit.ir.circuit import Circuit

_logger = logging.getLogger(__name__)


class SetModelPass(BasePass):
    """Sets a target machine model for future passes to target."""

    def __init__(self, model: MachineModel) -> None:
        """
        Construct a SetModelPass.

        Args:
            model (MachineModel | None): The machine model to encode
                into the compilation workflow.
        """
        if not isinstance(model, MachineModel):
            raise TypeError(f'Expected MachineModel, got: {type(model)}.')

        self.model = model

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        if 'machine_model' in data:
            _logger.warning('Overriding existing machine model.')

        if self.model.num_qudits < circuit.num_qudits:
            raise RuntimeError('Machine model is too small for circuit.')

        data['machine_model'] = self.model
