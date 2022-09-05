from __future__ import annotations

import cirq
import pytest
from cirq import Circuit as QuantumCircuit

from bqskit.compiler.compile import compile
from bqskit.ext import bqskit_to_cirq
from bqskit.ext import cirq_to_bqskit
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import HGate
from bqskit.qis import UnitaryMatrix


class TestTranslate:

    @pytest.fixture
    def bqskit_circuit(self) -> Circuit:
        circuit = Circuit(3)
        circuit.append_gate(CNOTGate(), (0, 1))
        circuit.append_gate(HGate(), 0)
        circuit.append_gate(HGate(), 1)
        circuit.append_gate(HGate(), 2)
        circuit.append_gate(CNOTGate(), (0, 1))
        circuit.append_gate(CNOTGate(), (0, 2))
        circuit.append_gate(CNOTGate(), (0, 2))
        circuit.append_gate(HGate(), 0)
        circuit.append_gate(HGate(), 1)
        circuit.append_gate(HGate(), 2)
        circuit.append_gate(HGate(), 2)
        circuit.append_gate(CNOTGate(), (0, 2))
        circuit.append_gate(CNOTGate(), (0, 2))
        circuit.append_gate(CNOTGate(), (0, 1))
        circuit.append_gate(CNOTGate(), (0, 2))
        circuit.append_gate(HGate(), 0)
        circuit.append_gate(HGate(), 1)
        circuit.append_gate(HGate(), 2)
        return circuit

    @pytest.fixture
    def cirq_circuit(self) -> QuantumCircuit:
        q0, q1, q2 = cirq.LineQubit.range(3)
        circuit = QuantumCircuit()
        circuit.append([cirq.CNOT(q0, q1)])
        circuit.append([cirq.H(q0)])
        circuit.append([cirq.H(q1)])
        circuit.append([cirq.H(q2)])
        circuit.append([cirq.CNOT(q0, q1)])
        circuit.append([cirq.CNOT(q0, q2)])
        circuit.append([cirq.CNOT(q0, q2)])
        circuit.append([cirq.H(q0)])
        circuit.append([cirq.H(q1)])
        circuit.append([cirq.H(q2)])
        circuit.append([cirq.H(q2)])
        circuit.append([cirq.CNOT(q0, q2)])
        circuit.append([cirq.CNOT(q0, q2)])
        circuit.append([cirq.CNOT(q0, q1)])
        circuit.append([cirq.CNOT(q0, q2)])
        circuit.append([cirq.H(q0)])
        circuit.append([cirq.H(q1)])
        circuit.append([cirq.H(q2)])
        return circuit

    def test_bqskit_to_bqskit(self, bqskit_circuit: Circuit) -> None:
        in_utry = bqskit_circuit.get_unitary()
        out_circuit = cirq_to_bqskit(bqskit_to_cirq(bqskit_circuit))
        out_utry = out_circuit.get_unitary()
        assert in_utry.get_distance_from(out_utry) < 1e-7

    def test_cirq_to_cirq(self, cirq_circuit: QuantumCircuit) -> None:
        qc = cirq_circuit
        in_utry = UnitaryMatrix(cirq.unitary(qc))
        out_circuit = bqskit_to_cirq(cirq_to_bqskit(qc))
        out_utry = UnitaryMatrix(cirq.unitary(out_circuit))
        assert in_utry.get_distance_from(out_utry) < 1e-7

    def test_compile_bqskit(self, cirq_circuit: QuantumCircuit) -> None:
        qc = cirq_circuit
        in_utry = UnitaryMatrix(cirq.unitary(qc))
        bqskit_circuit = cirq_to_bqskit(qc)
        bqskit_out_circuit = compile(bqskit_circuit, max_synthesis_size=2)
        out_circuit = bqskit_to_cirq(bqskit_out_circuit)
        out_utry = UnitaryMatrix(cirq.unitary(out_circuit))
        assert in_utry.get_distance_from(out_utry) < 1e-5

    def test_synthesis_bqskit(self, cirq_circuit: QuantumCircuit) -> None:
        qc = cirq_circuit
        in_utry = UnitaryMatrix(cirq.unitary(qc))
        bqskit_circuit = cirq_to_bqskit(qc)
        bqskit_out_circuit = compile(bqskit_circuit)
        out_circuit = bqskit_to_cirq(bqskit_out_circuit)
        out_utry = UnitaryMatrix(cirq.unitary(out_circuit))
        assert in_utry.get_distance_from(out_utry) < 1e-5

    def test_same_unitary(
        self,
        bqskit_circuit: Circuit,
        cirq_circuit: QuantumCircuit,
    ) -> None:
        bqskit_utry = bqskit_circuit.get_unitary()
        cirq_utry = UnitaryMatrix(cirq.unitary(cirq_circuit))
        assert bqskit_utry.get_distance_from(cirq_utry) < 1e-7
