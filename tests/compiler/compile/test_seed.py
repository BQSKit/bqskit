from __future__ import annotations

from random import random

from bqskit import compile
from bqskit.compiler.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import U3Gate


def test_same_seed_same_result(compiler: Compiler) -> None:
    circuit = Circuit(2)
    circuit.append_gate(U3Gate(), 0, [random() for _ in range(3)])
    circuit.append_gate(U3Gate(), 1, [random() for _ in range(3)])
    circuit.append_gate(CNOTGate(), (0, 1))
    circuit.append_gate(U3Gate(), 0, [random() for _ in range(3)])
    circuit.append_gate(U3Gate(), 1, [random() for _ in range(3)])
    circuit.append_gate(CNOTGate(), (0, 1))
    circuit.append_gate(U3Gate(), 0, [random() for _ in range(3)])
    circuit.append_gate(U3Gate(), 1, [random() for _ in range(3)])
    circuit.append_gate(CNOTGate(), (0, 1))
    circuit.append_gate(U3Gate(), 0, [random() for _ in range(3)])
    circuit.append_gate(U3Gate(), 1, [random() for _ in range(3)])
    circuit.append_gate(CNOTGate(), (0, 1))
    circuit.append_gate(U3Gate(), 0, [random() for _ in range(3)])
    circuit.append_gate(U3Gate(), 1, [random() for _ in range(3)])

    for _ in range(5):
        out_circuit1 = compile(circuit.copy(), seed=0)
        out_circuit2 = compile(circuit.copy(), seed=0)
        assert out_circuit1 == out_circuit2
