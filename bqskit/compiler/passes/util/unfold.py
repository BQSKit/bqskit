"""This module implements the UnfoldPass class."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.operation import Operation


_logger = logging.getLogger(__name__)


class UnfoldPass(BasePass):
    """
    The UnfoldPass class.

    The UnfoldPass unfolds all CircuitGate blocks into the circuit.
    """

    def run(self, circuit: Circuit, data: dict[str, Any]) -> None:
        """Perform the pass's operation, see BasePass for more info."""
        _logger.debug('Unfolding the circuit.')

        blocks: list[tuple[int, Operation]] = []
        for cycle, op in circuit.operations_with_cycles():
            if isinstance(op.gate, CircuitGate):
                blocks.append((cycle, op))

        prev_depth = circuit.get_depth()
        prev_cycle = prev_depth
        cycles_added = 0
        for cycle, op in sorted(blocks, key=lambda x: x[0], reverse=True):
            # Handle the case where the CircuitGate was moved due to inserts
            cycle_to_unfold = cycle + cycles_added if cycle == prev_cycle \
                else cycle
            circuit.unfold((cycle_to_unfold, op.location[0]))

            prev_cycle = cycle
            curr_depth = circuit.get_depth()
            cycles_added = curr_depth - prev_depth
            prev_depth = curr_depth
