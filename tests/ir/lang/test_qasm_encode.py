from __future__ import annotations

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import Reset
from bqskit.ir.gates import U3Gate
from bqskit.ir.lang.qasm2 import OPENQASM2Language
from bqskit.ext import qiskit_to_bqskit
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit


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
        expected = (
            'OPENQASM 2.0;\n'
            'include "qelib1.inc";\n'
            'qreg q[1];\n'
            'reset q[0];\n'
        )
        assert qasm == expected

    def test_classical_register(self) -> None:

        # Section: Circuit
        qr = QuantumRegister(2, name='qr')
        cr = ClassicalRegister(2, name='cr')
        qc = QuantumCircuit(qr, cr, name='qc')

        qc.h(qr[0])
        qc.cx(0, 1)

        qc.measure(qr, cr)

        bqc = qiskit_to_bqskit(qc)
        print(bqc)

        qasm = OPENQASM2Language().encode(bqc)
        expected = (
            'OPENQASM 2.0;\n'
            'include "qelib1.inc";\n'
            'qreg q[2];\n'
            'creg cr[2];\n'
            'h q[0];\n'
            'cx q[0], q[1];\n'
            'measure q[0] -> cr[0];\n'
            'measure q[1] -> cr[1];\n'
        )
        assert qasm == expected

    def test_multiple_classical_registers(self) -> None:
        # Section: Circuit
        qr = QuantumRegister(2, name='qr')
        cr1 = ClassicalRegister(2, name='cr1')
        cr2 = ClassicalRegister(2, name='cr2')
        qc = QuantumCircuit(qr, cr1, cr2, name='qc')

        qc.h(qr[0])
        qc.h(qr[1])
        qc.cx(qr[0], qr[1])

        qc.measure(qr, cr1)
        qc.measure(qr, cr2)

        bqc = qiskit_to_bqskit(qc)
        print(bqc)

        qasm = OPENQASM2Language().encode(bqc)
        expected = (
            'OPENQASM 2.0;\n'
            'include "qelib1.inc";\n'
            'qreg q[2];\n'
            'creg cr1[2];\n'
            'creg cr2[2];\n'
            'h q[0];\n'
            'h q[1];\n'
            'cx q[0], q[1];\n'
            'measure q[0] -> cr1[0];\n'
            'measure q[1] -> cr1[1];\n'
            'measure q[0] -> cr2[0];\n'
            'measure q[1] -> cr2[1];\n'
        )
        assert qasm == expected

    def test_gate_count(self) -> None:
        # Section: Circuit
        qr = QuantumRegister(2, name='qr')
        cr = ClassicalRegister(2, name='cr')
        qc = QuantumCircuit(qr, cr, name='qc')

        qc.h(qr[0])
        qc.h(qr[1])
        qc.cx(qr[0], qr[1])
        qc.measure(qr, cr)

        bqc = qiskit_to_bqskit(qc)
        print(bqc)

        qasm = OPENQASM2Language().encode(bqc)
        expected = (
            'OPENQASM 2.0;\n'
            'include "qelib1.inc";\n'
            'qreg q[2];\n'
            'creg cr[2];\n'
            'h q[0];\n'
            'h q[1];\n'
            'cx q[0], q[1];\n'
            'measure q[0] -> cr[0];\n'
            'measure q[1] -> cr[1];\n'
        )
        assert qasm == expected

        print({
            gate: bqc.count(gate)
            for gate in bqc.gate_set
        })
