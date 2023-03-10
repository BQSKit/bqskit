from __future__ import annotations

import pytest

from bqskit.compiler.machine import MachineModel
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gate import Gate
from bqskit.ir.gates import CXGate
from bqskit.ir.gates import HGate
from bqskit.ir.gates import XGate
from bqskit.passes.control.predicates.single import SinglePhysicalPredicate


@pytest.mark.parametrize(
    'gate_set', [
        set(),
        {HGate()},
        {XGate()},
        {HGate(), XGate()},
    ],
)
def test_single_physical(gate_set: set[Gate]) -> None:
    pred = SinglePhysicalPredicate()
    circuit = Circuit(2)
    circuit.append_gate(XGate(), 0)
    data = PassData(circuit)
    data.model = MachineModel(2, gate_set=gate_set)
    assert pred.get_truth_value(circuit, data) == (XGate() in gate_set)


def test_single_physical_mq_not_in() -> None:
    pred = SinglePhysicalPredicate()
    circuit = Circuit(2)
    circuit.append_gate(XGate(), 0)
    circuit.append_gate(CXGate(), (0, 1))
    data = PassData(circuit)
    data.model = MachineModel(2, gate_set={XGate()})
    assert pred.get_truth_value(circuit, data)


def test_single_physical_mq_in() -> None:
    pred = SinglePhysicalPredicate()
    circuit = Circuit(2)
    circuit.append_gate(XGate(), 0)
    circuit.append_gate(CXGate(), (0, 1))
    data = PassData(circuit)
    data.model = MachineModel(2, gate_set={XGate(), CXGate()})
    assert pred.get_truth_value(circuit, data)
