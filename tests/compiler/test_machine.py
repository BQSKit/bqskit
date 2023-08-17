from __future__ import annotations

import pytest

from bqskit.compiler.machine import MachineModel
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.constant.h import HGate
from bqskit.ir.gates.constant.t import TGate


class TestMachineConstructor:
    def test_coupling_graph(self) -> None:
        coupling_graph = {(0, 1), (1, 2), (2, 3)}
        t = MachineModel(4, coupling_graph)

        assert len(t.coupling_graph) == 3
        for link in coupling_graph:
            assert link in t.coupling_graph

    def test_num_qudits(self) -> None:
        for n in [1, 2, 3, 4]:
            t = MachineModel(n)
            assert t.num_qudits == n

    def test_alltoall_2(self) -> None:
        t = MachineModel(2)
        assert len(t.coupling_graph) == 1
        assert (0, 1) in t.coupling_graph

    def test_alltoall_3(self) -> None:
        t = MachineModel(3)
        assert len(t.coupling_graph) == 3
        assert (0, 1) in t.coupling_graph
        assert (0, 2) in t.coupling_graph
        assert (1, 2) in t.coupling_graph

    def test_alltoall_4(self) -> None:
        t = MachineModel(4)
        assert len(t.coupling_graph) == 6
        assert (0, 1) in t.coupling_graph
        assert (0, 2) in t.coupling_graph
        assert (0, 3) in t.coupling_graph
        assert (1, 2) in t.coupling_graph
        assert (1, 3) in t.coupling_graph
        assert (2, 3) in t.coupling_graph

    def test_num_qudits_invalid(self) -> None:
        with pytest.raises(TypeError):
            MachineModel('a')  # type: ignore
        with pytest.raises(ValueError):
            MachineModel(0)

    def test_coupling_graph_invalid(self) -> None:
        coupling_graph = {(0, 1), (1, 2), (2, 3)}
        with pytest.raises(TypeError):
            MachineModel(2, coupling_graph)
        with pytest.raises(TypeError):
            MachineModel(2, (0, 1))  # type: ignore
        with pytest.raises(TypeError):
            MachineModel(2, 0)  # type: ignore
        with pytest.raises(TypeError):
            MachineModel(2, 'a')  # type: ignore


def test_is_compatible() -> None:
    # Create a model with 3 qudits linearly connected
    model = MachineModel(
        num_qudits=3,
        coupling_graph=[(0, 1), (1, 2)],
        gate_set={HGate(), CNOTGate()},
        radixes=[2, 2, 2],
    )

    # Create a 2 qudit circuit with a gate set that is a subset
    circuit1 = Circuit(num_qudits=2, radixes=[2, 2])
    circuit1.append_gate(HGate(), 0)
    circuit1.append_gate(CNOTGate(), (0, 1))

    # Create a 3 qudit circuit with a gate set that is not a subset
    circuit2 = Circuit(num_qudits=3, radixes=[2, 2, 2])
    circuit2.append_gate(HGate(), 0)
    circuit2.append_gate(TGate(), 1)
    circuit2.append_gate(CNOTGate(), (1, 2))

    # Create a 3 qudit circuit with a different radix
    circuit3 = Circuit(num_qudits=3, radixes=[2, 2, 3])
    circuit3.append_gate(HGate(), 0)
    circuit3.append_gate(CNOTGate(), (0, 1))

    # Create a 3 qudit circuit with a different coupling graph
    circuit4 = Circuit(num_qudits=3, radixes=[2, 2, 2])
    circuit4.append_gate(HGate(), 0)
    circuit4.append_gate(CNOTGate(), (0, 2))

    # Create a 3 qudit circuit with a valid placement
    circuit5 = Circuit(num_qudits=3, radixes=[2, 2, 2])
    circuit5.append_gate(HGate(), 0)
    circuit5.append_gate(CNOTGate(), (0, 2))
    placement = [0, 2, 1]

    # Check compatibility of each circuit with the model
    assert model.is_compatible(circuit1)
    assert not model.is_compatible(circuit2)
    assert not model.is_compatible(circuit3)
    assert not model.is_compatible(circuit4)
    assert model.is_compatible(circuit5, placement)
