from __future__ import annotations

import pytest

from bqskit.compiler.machine import MachineModel
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gate import Gate
from bqskit.ir.gates import CXGate
from bqskit.ir.gates import HGate
from bqskit.ir.gates import XGate
from bqskit.passes.control.predicates import PhysicalPredicate


@pytest.mark.parametrize(
    'gate_set', [
        set(),
        {HGate()},
        {XGate()},
        {HGate(), XGate()},
    ],
)
def test_physical(gate_set: set[Gate]) -> None:
    pred = PhysicalPredicate()
    circuit = Circuit(2)
    circuit.append_gate(XGate(), 0)
    data = PassData(circuit)
    data.model = MachineModel(2, gate_set=gate_set)
    assert pred.get_truth_value(circuit, data) == (XGate() in gate_set)


def test_physical_mq_not_in() -> None:
    pred = PhysicalPredicate()
    circuit = Circuit(2)
    circuit.append_gate(XGate(), 0)
    circuit.append_gate(CXGate(), (0, 1))
    data = PassData(circuit)
    data.model = MachineModel(2, gate_set={XGate()})
    assert not pred.get_truth_value(circuit, data)


def test_physical_mq_in() -> None:
    pred = PhysicalPredicate()
    circuit = Circuit(2)
    circuit.append_gate(XGate(), 0)
    circuit.append_gate(CXGate(), (0, 1))
    data = PassData(circuit)
    data.model = MachineModel(2, gate_set={XGate(), CXGate()})
    assert pred.get_truth_value(circuit, data)
