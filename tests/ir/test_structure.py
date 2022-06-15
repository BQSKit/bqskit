"""This module tests the CircuitStructure class."""
from __future__ import annotations

from bqskit.ir import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import U3Gate
from bqskit.ir.structure import CircuitStructure


def test_structure_const() -> None:
    circuit = Circuit(2)
    circuit.append_gate(CNOTGate(), (0, 1))
    circuit.append_gate(U3Gate(), 0)
    circuit.append_gate(U3Gate(), 1)
    structure = CircuitStructure(circuit)

    expected = (
        ('CNOTGate@(0, 1)', 'CNOTGate@(0, 1)'),
        ('U3Gate@(0,)', 'U3Gate@(1,)'),
    )

    assert structure.structure == expected


def test_structure_compressed() -> None:
    circuit1 = Circuit(2)
    circuit1.append_gate(CNOTGate(), (0, 1))
    circuit1.append_gate(U3Gate(), 0)
    circuit1.append_gate(U3Gate(), 1)
    structure1 = CircuitStructure(circuit1)

    circuit2 = Circuit(2)
    circuit2.append_gate(CNOTGate(), (0, 1))
    circuit2.append_gate(U3Gate(), 0)
    circuit2.append_gate(CNOTGate(), (0, 1))
    circuit2.append_gate(U3Gate(), 1)
    circuit2.pop((2, 0))
    structure2 = CircuitStructure(circuit2)

    assert structure1 == structure2
    assert hash(structure1) == hash(structure2)
