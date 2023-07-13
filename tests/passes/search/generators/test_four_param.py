from __future__ import annotations

from bqskit.ir import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import RXGate
from bqskit.ir.gates import RYGate
from bqskit.ir.gates import RZGate
from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.passes.search.generators import FourParamGenerator


def test_fringe_cnot_count_empty_circuit() -> None:
    gen = FourParamGenerator()
    assert gen.count_outer_cnots(Circuit(1), (0, 1)) == 0


def test_fringe_cnot_count_2q_circuit() -> None:
    gen = FourParamGenerator()
    circuit = Circuit(2)
    circuit.append_gate(U3Gate(), 0)
    circuit.append_gate(U3Gate(), 1)
    for i in range(1, 10):
        circuit.append_gate(CNOTGate(), (0, 1))
        circuit.append_gate(RYGate(), 0)
        circuit.append_gate(RZGate(), 0)
        circuit.append_gate(RYGate(), 1)
        circuit.append_gate(RXGate(), 1)
        assert gen.count_outer_cnots(circuit, (0, 1)) == i


def test_fringe_cnot_count_4q_circuit() -> None:
    gen = FourParamGenerator()
    circuit = Circuit(4)
    for i in range(4):
        circuit.append_gate(U3Gate(), i)

    circuit.append_gate(CNOTGate(), (0, 1))
    circuit.append_gate(RYGate(), 0)
    circuit.append_gate(RZGate(), 0)
    circuit.append_gate(RYGate(), 1)
    circuit.append_gate(RXGate(), 1)
    circuit.append_gate(CNOTGate(), (2, 3))
    circuit.append_gate(RYGate(), 2)
    circuit.append_gate(RZGate(), 2)
    circuit.append_gate(RYGate(), 3)
    circuit.append_gate(RXGate(), 3)

    for i in range(1, 10):
        circuit.append_gate(CNOTGate(), (1, 2))
        circuit.append_gate(RYGate(), 1)
        circuit.append_gate(RZGate(), 1)
        circuit.append_gate(RYGate(), 2)
        circuit.append_gate(RXGate(), 2)
        assert gen.count_outer_cnots(circuit, (1, 2)) == i


def test_fringe_cnot_count_3q_hidden() -> None:
    gen = FourParamGenerator()
    circuit = Circuit(3)
    for i in range(3):
        circuit.append_gate(U3Gate(), i)

    circuit.append_gate(CNOTGate(), (0, 1))
    circuit.append_gate(RYGate(), 0)
    circuit.append_gate(RZGate(), 0)
    circuit.append_gate(RYGate(), 1)
    circuit.append_gate(RXGate(), 1)
    circuit.append_gate(CNOTGate(), (0, 1))
    circuit.append_gate(RYGate(), 0)
    circuit.append_gate(RZGate(), 0)
    circuit.append_gate(RYGate(), 1)
    circuit.append_gate(RXGate(), 1)
    circuit.append_gate(CNOTGate(), (0, 1))
    circuit.append_gate(RYGate(), 0)
    circuit.append_gate(RZGate(), 0)
    circuit.append_gate(RYGate(), 1)
    circuit.append_gate(RXGate(), 1)
    circuit.append_gate(CNOTGate(), (1, 2))
    circuit.append_gate(RYGate(), 1)
    circuit.append_gate(RZGate(), 1)
    circuit.append_gate(RYGate(), 2)
    circuit.append_gate(RXGate(), 2)
    assert gen.count_outer_cnots(circuit, (0, 1)) == 0
