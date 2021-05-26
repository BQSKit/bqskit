"""This module implements the UnfoldPass class."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.operation import Operation
from bqskit.ir.point import CircuitPoint


_logger = logging.getLogger(__name__)


class UnfoldPass(BasePass):
    """
    The UnfoldPass class.

    The UnfoldPass unfolds all CircuitGate blocks into the circuit.
    """

    def run(self, circuit: Circuit, data: dict[str, Any]) -> None:
        """Perform the pass's operation, see BasePass for more info."""
        _logger.debug('Unfolding the circuit.')

        blocks: list[tuple[CircuitPoint, Operation]] = []
        for point, op in circuit.operations_with_points():
            if isinstance(op.gate, CircuitGate):
                blocks.append((point, op))

        for point, op in sorted(blocks, key=lambda x: x[0], reverse=True):
            circuit.unfold(point)
