from __future__ import annotations

import pytest
from qutip import CircuitSimulator
from qutip import QubitCircuit

from bqskit.compiler.compile import compile
from bqskit.ext import bqskit_to_qutip
from bqskit.ext import qutip_to_bqskit
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import U3Gate
from bqskit.qis import UnitaryMatrix


class TestTranslate:

    @pytest.fixture
    def bqskit_circuit(self) -> Circuit:
        circuit = Circuit(3)
        circuit.append_gate(CNOTGate(), (0, 1))
        circuit.append_gate(U3Gate(), 0, [1, 2, 3])
        circuit.append_gate(U3Gate(), 1, [1, 2, 3])
        circuit.append_gate(U3Gate(), 2, [1, 2, 3])
        circuit.append_gate(CNOTGate(), (0, 1))
        circuit.append_gate(CNOTGate(), (0, 2))
        circuit.append_gate(CNOTGate(), (0, 2))
        circuit.append_gate(U3Gate(), 0, [1, 2.4, 3])
        circuit.append_gate(U3Gate(), 1, [1, 2.2, 3])
        circuit.append_gate(U3Gate(), 2, [1, 2.1, 3])
        circuit.append_gate(U3Gate(), 2, [1, 2.1, 3])
        circuit.append_gate(CNOTGate(), (0, 2))
        circuit.append_gate(CNOTGate(), (0, 2))
        circuit.append_gate(CNOTGate(), (0, 1))
        circuit.append_gate(CNOTGate(), (0, 2))
        circuit.append_gate(U3Gate(), 0, [1, 2.4, 3])
        circuit.append_gate(U3Gate(), 1, [1, 2.2, 3])
        circuit.append_gate(U3Gate(), 2, [1, 2.1, 3])
        return circuit

    @pytest.fixture
    def qutip_circuit(self) -> QubitCircuit:
        circuit = QubitCircuit(3)
        circuit.add_gate('CNOT', controls=0, targets=1)
        circuit.add_gate('QASMU', targets=0, arg_value=[1, 2, 3])
        circuit.add_gate('QASMU', targets=1, arg_value=[1, 2, 3])
        circuit.add_gate('QASMU', targets=2, arg_value=[1, 2, 3])
        circuit.add_gate('CNOT', controls=0, targets=1)
        circuit.add_gate('CNOT', controls=0, targets=2)
        circuit.add_gate('CNOT', controls=0, targets=2)
        circuit.add_gate('QASMU', targets=0, arg_value=[1, 2.4, 3])
        circuit.add_gate('QASMU', targets=1, arg_value=[1, 2.2, 3])
        circuit.add_gate('QASMU', targets=2, arg_value=[1, 2.1, 3])
        circuit.add_gate('QASMU', targets=2, arg_value=[1, 2.1, 3])
        circuit.add_gate('CNOT', controls=0, targets=2)
        circuit.add_gate('CNOT', controls=0, targets=2)
        circuit.add_gate('CNOT', controls=0, targets=1)
        circuit.add_gate('CNOT', controls=0, targets=2)
        circuit.add_gate('QASMU', targets=0, arg_value=[1, 2.4, 3])
        circuit.add_gate('QASMU', targets=1, arg_value=[1, 2.2, 3])
        circuit.add_gate('QASMU', targets=2, arg_value=[1, 2.1, 3])
        return circuit

    def test_bqskit_to_bqskit(self, bqskit_circuit: Circuit) -> None:
        in_utry = bqskit_circuit.get_unitary()
        out_circuit = qutip_to_bqskit(bqskit_to_qutip(bqskit_circuit))
        out_utry = out_circuit.get_unitary()
        assert in_utry.get_distance_from(out_utry) < 1e-7

    def test_qutip_to_qutip(self, qutip_circuit: QubitCircuit) -> None:
        qc = qutip_circuit
        utry = CircuitSimulator(qc, precompute_unitary=True).ops[0].data
        in_utry = UnitaryMatrix(utry.todense())
        oc = bqskit_to_qutip(qutip_to_bqskit(qc))
        utry = CircuitSimulator(oc, precompute_unitary=True).ops[0].data
        out_utry = UnitaryMatrix(utry.todense())
        assert in_utry.get_distance_from(out_utry) < 1e-7

    def test_compile_bqskit(self, qutip_circuit: QubitCircuit) -> None:
        qc = qutip_circuit
        utry = CircuitSimulator(qc, precompute_unitary=True).ops[0].data
        in_utry = UnitaryMatrix(utry.todense())
        bqskit_circuit = qutip_to_bqskit(qc)
        bqskit_out_circuit = compile(bqskit_circuit, max_synthesis_size=2)
        oc = bqskit_to_qutip(bqskit_out_circuit)
        utry = CircuitSimulator(oc, precompute_unitary=True).ops[0].data
        out_utry = UnitaryMatrix(utry.todense())
        assert in_utry.get_distance_from(out_utry) < 1e-5

    def test_synthesis_bqskit(self, qutip_circuit: QubitCircuit) -> None:
        qc = qutip_circuit
        utry = CircuitSimulator(qc, precompute_unitary=True).ops[0].data
        in_utry = UnitaryMatrix(utry.todense())
        bqskit_circuit = qutip_to_bqskit(qc)
        bqskit_out_circuit = compile(bqskit_circuit)
        oc = bqskit_to_qutip(bqskit_out_circuit)
        utry = CircuitSimulator(oc, precompute_unitary=True).ops[0].data
        out_utry = UnitaryMatrix(utry.todense())
        assert in_utry.get_distance_from(out_utry) < 1e-5

    def test_same_unitary(
        self,
        bqskit_circuit: Circuit,
        qutip_circuit: QubitCircuit,
    ) -> None:
        bqskit_utry = bqskit_circuit.get_unitary()
        qc = qutip_circuit
        utry = CircuitSimulator(qc, precompute_unitary=True).ops[0].data
        qutip_utry = UnitaryMatrix(utry.todense())
        assert bqskit_utry.get_distance_from(qutip_utry) < 1e-7
