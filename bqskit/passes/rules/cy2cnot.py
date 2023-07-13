"""This module implements the CYToCNOTPass."""
from __future__ import annotations

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CircuitGate
from bqskit.ir.gates.constant import CNOTGate
from bqskit.ir.gates.constant import CYGate
from bqskit.ir.gates.constant import SdgGate
from bqskit.ir.gates.constant import SGate
from bqskit.ir.operation import Operation


class CYToCNOTPass(BasePass):
    """
    The CYToCNOTPass class.

    This uses a rule to convert CYs to CNOTs.
    """

    def __init__(self) -> None:
        """Construct a CYToCNOTPass."""
        circuit = Circuit(2)
        circuit.append_gate(SdgGate(), 1)
        circuit.append_gate(CNOTGate(), (0, 1))
        circuit.append_gate(SGate(), 1)
        self.cg = CircuitGate(circuit)

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""

        # Find all cnots
        cnot_points = []
        for cycle, op in circuit.operations_with_cycles():
            if isinstance(op.gate, CYGate):
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
