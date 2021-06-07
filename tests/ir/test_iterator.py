"""This module tests the CircuitIterator class."""
from __future__ import annotations

from bqskit.ir import Circuit
from bqskit.ir import CircuitIterator
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import HGate


class TestCircuitIterator:

    def test_empty(self) -> None:
        circuit = Circuit(1)
        ops = [op for op in CircuitIterator(circuit)]
        assert len(ops) == 0

    def test_one_gate_1(self) -> None:
        circuit = Circuit(1)
        circuit.append_gate(HGate(), 0)
        ops = [op for op in CircuitIterator(circuit)]
        assert len(ops) == 1
        assert ops[0].gate == HGate()  # type: ignore

    def test_one_gate_2(self) -> None:
        circuit = Circuit(2)
        circuit.append_gate(CNOTGate(), (0, 1))
        ops = [op for op in CircuitIterator(circuit)]
        assert len(ops) == 1
        assert ops[0].gate == CNOTGate()  # type: ignore

    def test_one_gate_3(self) -> None:
        circuit = Circuit(3)
        circuit.append_gate(CNOTGate(), (0, 1))
        ops = [op for op in CircuitIterator(circuit)]
        assert len(ops) == 1
        assert ops[0].gate == CNOTGate()  # type: ignore

    def test_one_gate_4(self) -> None:
        circuit = Circuit(3)
        circuit.append_gate(CNOTGate(), (1, 2))
        ops = [op for op in CircuitIterator(circuit)]
        assert len(ops) == 1
        assert ops[0].gate == CNOTGate()  # type: ignore
