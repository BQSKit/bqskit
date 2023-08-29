"""This module implements the GateCountPredicate class."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from bqskit.ir.gate import Gate
from bqskit.passes.control.predicate import PassPredicate
from bqskit.utils.typing import is_sequence

if TYPE_CHECKING:
    from typing import Sequence

    from bqskit.compiler.passdata import PassData
    from bqskit.ir.circuit import Circuit


_logger = logging.getLogger(__name__)


class GateCountPredicate(PassPredicate):
    """
    The GateCountPredicate class.

    The GateCountPredicate returns true if the number of one of more types of
    gate has changed in the circuit.
    """

    key = 'GateCountPredicate_circuit_count'

    def __init__(self, gate: Gate | Sequence[Gate] | str) -> None:
        """
        Construct a GateCountPredicate.

        Args:
            gate (Gate | Sequence[Gate] | str): The gate(s) to count.
             Either a single gate, a sequence of gates, or a string
                representing the gate type. The allowed strings are:
                    - 'sq' for single qudit gates
                    - 'tq' for two qudit gates
                    - 'multi' for multi-qudit gates
                    - 'many' for gates with more than 2 qudits
        """

        if isinstance(gate, Gate):
            gate = [gate]

        if not isinstance(gate, str):
            if not is_sequence(gate):
                raise TypeError(f'Expected gate, got {type(gate)}')

            if not all(isinstance(g, Gate) for g in gate):
                raise TypeError(f'Expected gate, got {type(gate)}')

        else:
            if gate not in ['sq', 'tq', 'multi', 'many']:
                raise ValueError(f'Unknown gate type {gate}.')

        self.gate = list(gate) if not isinstance(gate, str) else gate

    def get_truth_value(self, circuit: Circuit, data: PassData) -> bool:
        """Call this predicate, see :class:`PassPredicate` for more info."""
        if isinstance(self.gate, str):
            gates = list({
                'sq': circuit.gate_set.single_qudit_gates,
                'tq': circuit.gate_set.two_qudit_gates,
                'multi': circuit.gate_set.multi_qudit_gates,
                'many': circuit.gate_set.many_qudit_gates,
            }[self.gate])

        else:
            gates = self.gate  # type: ignore

        gate_count = sum(circuit.count(g) for g in gates)

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
