from __future__ import annotations

import pytest

from bqskit.compiler.machine import MachineModel
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gate import Gate
from bqskit.ir.gates import CXGate
from bqskit.ir.gates import CZGate
from bqskit.ir.gates import XGate
from bqskit.passes.control.predicates.multi import MultiPhysicalPredicate


@pytest.mark.parametrize(
    'gate_set', [
        set(),
        {CZGate()},
        {CXGate()},
        {CZGate(), CXGate()},
    ],
)
def test_multi_physical(gate_set: set[Gate]) -> None:
    pred = MultiPhysicalPredicate()
    circuit = Circuit(2)
    circuit.append_gate(CXGate(), (0, 1))
    data = PassData(circuit)
    data.model = MachineModel(2, gate_set=gate_set)
    assert pred.get_truth_value(circuit, data) == (CXGate() in gate_set)


def test_multi_physical_sq_not_in() -> None:
    pred = MultiPhysicalPredicate()
    circuit = Circuit(2)
    circuit.append_gate(XGate(), 0)
    circuit.append_gate(CXGate(), (0, 1))
    data = PassData(circuit)
    data.model = MachineModel(2, gate_set={CXGate()})
    assert pred.get_truth_value(circuit, data)


def test_multi_physical_mq_in() -> None:
    pred = MultiPhysicalPredicate()
    circuit = Circuit(2)
    circuit.append_gate(XGate(), 0)
    circuit.append_gate(CXGate(), (0, 1))
    data = PassData(circuit)
    data.model = MachineModel(2, gate_set={XGate(), CXGate()})
    assert pred.get_truth_value(circuit, data)
