"""This module implements the UnfoldPass class."""
from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit


_logger = logging.getLogger(__name__)


class UnfoldPass(BasePass):
    """
    The UnfoldPass class.

    The UnfoldPass unfolds all CircuitGate blocks into the circuit.
    """

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        _logger.debug('Unfolding the circuit.')
        circuit.unfold_all()
