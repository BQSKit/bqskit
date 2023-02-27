from __future__ import annotations

from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CXGate
from bqskit.ir.gates import HGate
from bqskit.passes.control.predicates import ChangePredicate


def test_change_single_gate() -> None:
    pred = ChangePredicate()
    circuit = Circuit(2)
    data = PassData(circuit)

    assert pred.get_truth_value(circuit, data)

    for _ in range(3):
        assert not pred.get_truth_value(circuit, data)

    for _ in range(3):
        circuit.append_gate(CXGate(), (0, 1))

        assert pred.get_truth_value(circuit, data)
        for _ in range(3):
            assert not pred.get_truth_value(circuit, data)


def test_change_two_gate() -> None:
    pred = ChangePredicate()
    circuit = Circuit(2)
    data = PassData(circuit)

    assert pred.get_truth_value(circuit, data)

    for _ in range(3):
        assert not pred.get_truth_value(circuit, data)

    for _ in range(3):
        circuit.append_gate(CXGate(), (0, 1))

        assert pred.get_truth_value(circuit, data)
        for _ in range(3):
            assert not pred.get_truth_value(circuit, data)

        circuit.append_gate(HGate(), 0)

        assert pred.get_truth_value(circuit, data)
        for _ in range(3):
            assert not pred.get_truth_value(circuit, data)


def test_change_two_gate_plus_minus() -> None:
    pred = ChangePredicate()
    circuit = Circuit(2)
    data = PassData(circuit)

    circuit.append_gate(HGate(), 0)
    circuit.append_gate(CXGate(), (0, 1))

    assert pred.get_truth_value(circuit, data)

    circuit.remove(CXGate())
    circuit.append_gate(HGate(), 0)

    assert pred.get_truth_value(circuit, data)
