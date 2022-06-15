"""This module implements the SwapToCNOTPass."""
from __future__ import annotations

from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CircuitGate
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import SwapGate
from bqskit.ir.operation import Operation


class SwapToCNOTPass(BasePass):
    """
    The SwapToCNOTPass class.

    This uses a rule to convert Swaps to CNOTs.
    """

    def __init__(self) -> None:
        """Construct a SwapToCNOTPass."""
        circuit = Circuit(2)
        circuit.append_gate(CNOTGate(), (1, 0))
        circuit.append_gate(CNOTGate(), (0, 1))
        circuit.append_gate(CNOTGate(), (1, 0))
        self.cg = CircuitGate(circuit)

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""

        # Find all cnots
        cnot_points = []
        for cycle, op in circuit.operations_with_cycles():
            if isinstance(op.gate, SwapGate):
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
