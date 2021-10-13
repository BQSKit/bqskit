"""This module implements the ToU3Pass."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.parameterized.pauli import PauliGate
from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.ir.point import CircuitPoint

_logger = logging.getLogger(__name__)


class ToU3Pass(BasePass):
    """Converts single-qubit general unitary gates to U3 Gates."""

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        _logger.debug('Converting single-qubit general gates to U3Gates.')
        for cycle, op in circuit.operations_with_cycles():
            if (
                (
                    isinstance(op.gate, VariableUnitaryGate)
                    or isinstance(op.gate, PauliGate)
                )
                and len(op.location) == 1
                and op.radixes == (2,)
            ):
                params = U3Gate.calc_params(op.get_unitary())
                point = CircuitPoint(cycle, op.location[0])
                circuit.replace_gate(point, U3Gate(), op.location, params)
