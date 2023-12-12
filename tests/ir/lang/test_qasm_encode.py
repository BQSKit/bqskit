from __future__ import annotations

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import U3Gate
from bqskit.ir.gates import Reset
from bqskit.ir.lang.qasm2 import OPENQASM2Language


class TestCircuitGates:
    def test_circuitgate(self) -> None:
        subcircuit = Circuit(4)
        subcircuit.append_gate(CNOTGate(), (0, 1))
        subcircuit.append_gate(U3Gate(), 0, [1, 2, 3])
        subcircuit.append_gate(U3Gate(), 0, [1, 2, 3])
        circuit = Circuit(4)
        circuit.append_circuit(subcircuit, (0, 1, 2, 3), True)
        circuit.append_circuit(subcircuit, (0, 1, 2, 3), True)
        in_utry = circuit.get_unitary()

        qasm = OPENQASM2Language().encode(circuit)
        parsed_circuit = OPENQASM2Language().decode(qasm)
        assert parsed_circuit.get_unitary().get_distance_from(in_utry) < 1e-7

    def test_nested_circuitgate(self) -> None:
        subsubcircuit = Circuit(4)
        subsubcircuit.append_gate(CNOTGate(), (0, 1))
        subsubcircuit.append_gate(U3Gate(), 0, [1, 2, 3])
        subsubcircuit.append_gate(U3Gate(), 0, [1, 2, 3])
        subcircuit1 = Circuit(4)
        subcircuit1.append_circuit(subsubcircuit, (0, 1, 2, 3), True)
        subcircuit1.append_circuit(subsubcircuit, (0, 1, 2, 3), True)
        subcircuit2 = Circuit(2)
        subcircuit2.append_gate(U3Gate(), 0, [4, 6, 1.2])
        subcircuit2.append_gate(U3Gate(), 1, [4, 6, 1.2])
        circuit = Circuit(4)
        circuit.append_circuit(subcircuit1, (0, 1, 2, 3), True)
        circuit.append_circuit(subcircuit2, (0, 3), True)
        circuit.append_circuit(subcircuit1, (0, 1, 2, 3), True)
        in_utry = circuit.get_unitary()

        qasm = OPENQASM2Language().encode(circuit)
        parsed_circuit = OPENQASM2Language().decode(qasm)
        assert parsed_circuit.get_unitary().get_distance_from(in_utry) < 1e-7

    def test_reset(self) -> None:
        circuit = Circuit(1)
        circuit.append_gate(Reset(), 0)
        qasm = OPENQASM2Language().encode(circuit)

        expected = """
            OPENQASM 2.0;
            qreg q[1];
            reset q[0];
        """
        assert qasm == expected

