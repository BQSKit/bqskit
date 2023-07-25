"""This module implements the ToU3Pass."""
from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.ir.point import CircuitPoint

_logger = logging.getLogger(__name__)


class FromU3ToVariablePass(BasePass):
    """Converts U3 Gates to single-qubit variable unitary gates."""

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        _logger.debug('Converting U3Gates to VariableUnitaryGate')
        for cycle, op in circuit.operations_with_cycles():
            if isinstance(op.gate, U3Gate):
                params = VariableUnitaryGate.get_params(op.get_unitary())
                point = CircuitPoint(cycle, op.location[0])
                vgate = VariableUnitaryGate(op.num_qudits, op.radixes)
                circuit.replace_gate(
                    point, vgate, op.location, params,
                )
