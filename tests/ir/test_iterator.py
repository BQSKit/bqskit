"""This module tests the CircuitIterator class."""
from __future__ import annotations

from typing import cast

from hypothesis import given

from bqskit.ir import CircuitIterator
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import HGate
from bqskit.ir.operation import Operation
from bqskit.utils.test.strategies import circuits


class TestCircuitIterator:
    def test_empty(self) -> None:
        circuit = Circuit(1)
        ops = [cast(Operation, op) for op in CircuitIterator(circuit)]
        assert len(ops) == 0

    def test_one_gate_1(self) -> None:
        circuit = Circuit(1)
        circuit.append_gate(HGate(), 0)
        ops = [cast(Operation, op) for op in CircuitIterator(circuit)]
        assert len(ops) == 1
        assert ops[0].gate == HGate()

    def test_one_gate_2(self) -> None:
        circuit = Circuit(2)
        circuit.append_gate(CNOTGate(), (0, 1))
        ops = [cast(Operation, op) for op in CircuitIterator(circuit)]
        assert len(ops) == 1
        assert ops[0].gate == CNOTGate()

    def test_one_gate_3(self) -> None:
        circuit = Circuit(3)
        circuit.append_gate(CNOTGate(), (0, 1))
        ops = [cast(Operation, op) for op in CircuitIterator(circuit)]
        assert len(ops) == 1
        assert ops[0].gate == CNOTGate()

    def test_one_gate_4(self) -> None:
        circuit = Circuit(3)
        circuit.append_gate(CNOTGate(), (1, 2))
        ops = [cast(Operation, op) for op in CircuitIterator(circuit)]
        assert len(ops) == 1
        assert ops[0].gate == CNOTGate()

    @given(circuits((2, 2, 2)))
    def test_reconstruct_circuit(self, circuit: Circuit) -> None:
        new_circuit = Circuit(circuit.num_qudits, circuit.radixes)
        for op in CircuitIterator(circuit):
            new_circuit.append(cast(Operation, op))
        assert new_circuit.get_unitary() == circuit.get_unitary()
