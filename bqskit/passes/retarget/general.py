"""This module implements the GeneralSQDecomposition."""
from __future__ import annotations

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gate import Gate
from bqskit.ir.gates.generalgate import GeneralGate
from bqskit.ir.gates.parameterized.u3 import U3Gate


class GeneralSQDecomposition(BasePass):
    """Convert to a general single-qudit gate from the model's gate set."""

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""

        if circuit.num_qudits != 1:
            raise ValueError(
                'Cannot convert multi-qudit circuit into'
                ' general single-qudit gate.',
            )
        
        # Check radixes
        radix = circuit.radixes[0]
        
        # Pick a general gate
        general_gate: Gate | None = None

        for gate in data.gate_set.single_qudit_gates:
            if isinstance(gate, GeneralGate):
                if gate.radixes[0] == radix:
                    general_gate = gate
                    break
        
        if general_gate is None:
            raise ValueError(
                f'No general single-qudit gate with radix {radix} in gate set.',
            )

        utry = circuit.get_unitary()
        new_circuit = Circuit(1)
        new_circuit.append_gate(general_gate, 0, general_gate.calc_params(utry))
        circuit.become(new_circuit)
