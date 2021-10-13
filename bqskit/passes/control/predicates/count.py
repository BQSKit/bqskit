"""This module implements the GateCountPredicate class."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.ir.circuit import Circuit
from bqskit.ir.gate import Gate
from bqskit.passes.control.predicate import PassPredicate

_logger = logging.getLogger(__name__)


class GateCountPredicate(PassPredicate):
    """
    The GateCountPredicate class.

    The GateCountPredicate returns true if the number of one type of gate has
    changed in the circuit.
    """

    key = 'GateCountPredicate_circuit_count'

    def __init__(self, gate: Gate) -> None:
        """Construct a GateCountPredicate."""

        if not isinstance(gate, Gate):
            raise TypeError(f'Expected gate, got {type(gate)}')

        self.gate = gate

    def get_truth_value(self, circuit: Circuit, data: dict[str, Any]) -> bool:
        """Call this predicate, see :class:`PassPredicate` for more info."""
        gate_count = circuit.count(self.gate)

        # If first call, record data and return true
        if self.key not in data:
            _logger.debug(f'Could not find {self.key} key.')
            data[self.key] = gate_count
            return True

        # Otherwise, check for a change and return accordingly
        if data[self.key] == gate_count:
            _logger.debug('Counts match; no change detected.')
            return False

        data[self.key] = gate_count
        _logger.debug('Counts do not match; change detected.')
        return True
