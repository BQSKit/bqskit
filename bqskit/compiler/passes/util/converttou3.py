"""This module converts VariableUnitary Gates into U3 gates."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.parameterized.pauli import PauliGate
from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.ir.point import CircuitPoint

_logger = logging.getLogger(__name__)


class VariableToU3Pass(BasePass):
    """Convert VariableUnitaryGates to U3 Gates."""

    def run(self, circuit: Circuit, data: dict[str, Any]) -> None:
        """Perform the pass's operation, see BasePass for more info."""
        _logger.debug('Converting VariableUnitaryGates to U3Gates.')
        for cycle, op in circuit.operations_with_cycles():
            if isinstance(op.gate, VariableUnitaryGate) and len(
                    op.location,
            ) == 1:
                # Convert to SU(2)
                unitary = op.get_unitary().get_numpy()
                mag = np.linalg.det(unitary) ** (-1 / 2)
                special_unitary = mag * unitary
                a = np.angle(special_unitary[1, 1])
                b = np.angle(special_unitary[1, 0])
                # Get angles
                theta = float(
                    np.arctan2(
                        np.abs(special_unitary[1, 0]),
                        np.abs(special_unitary[0, 0]),
                    ),
                ) * 2
                phi = (a + b)
                lamb = (a - b)
                # Replace
                point = CircuitPoint(cycle, op.location[0])
                circuit.replace_gate(
                    point, U3Gate(), op.location,
                    [theta, phi, lamb],
                )


class PauliToU3Pass(BasePass):
    """Convert PauliGates to U3 Gates."""

    def run(self, circuit: Circuit, data: dict[str, Any]) -> None:
        """Perform the pass's operation, see BasePass for more info."""
        _logger.debug('Converting PauliGates to U3Gates.')
        for cycle, op in circuit.operations_with_cycles():
            if isinstance(op.gate, PauliGate) and len(
                    op.location,
            ) == 1:
                # Convert to SU(2)
                unitary = op.get_unitary().get_numpy()
                mag = np.linalg.det(unitary) ** (-1 / 2)
                special_unitary = mag * unitary
                a = np.angle(special_unitary[1, 1])
                b = np.angle(special_unitary[1, 0])
                # Get angles
                theta = float(
                    np.arctan2(
                        np.abs(special_unitary[1, 0]),
                        np.abs(special_unitary[0, 0]),
                    ),
                ) * 2
                phi = (a + b)
                lamb = (a - b)
                # Replace
                point = CircuitPoint(cycle, op.location[0])
                circuit.replace_gate(
                    point, U3Gate(), op.location,
                    [theta, phi, lamb],
                )
