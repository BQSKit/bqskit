"""This module implements the CHToCNOTPass."""
from __future__ import annotations

from typing import Any

import numpy as np

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CHGate
from bqskit.ir.gates import CircuitGate
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import RYGate
from bqskit.ir.operation import Operation


class CHToCNOTPass(BasePass):
    """
    The CHToCNOTPass class.

    This uses a rule to convert CHs to CNOTs.
    """

    def __init__(self) -> None:
        """Construct a CHToCNOTPass."""
        circuit = Circuit(2)
        circuit.append_gate(RYGate(), 1, [np.pi / 4])
        circuit.append_gate(CNOTGate(), (0, 1))
        circuit.append_gate(RYGate(), 1, [-np.pi / 4])
        self.cg = CircuitGate(circuit)

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""

        # Find all cnots
        cnot_points = []
        for cycle, op in circuit.operations_with_cycles():
            if isinstance(op.gate, CHGate):
                cnot_points.append((cycle, op.location[0]))

        # Create new operations
        ops = [
            Operation(
                self.cg,
                circuit[p].location,
                self.cg._circuit.params,
            )
            for p in cnot_points
        ]

        # Replace cnots with new ops
        circuit.batch_replace(cnot_points, ops)
        circuit.unfold_all()  # TODO: Replace with batch_unfold
