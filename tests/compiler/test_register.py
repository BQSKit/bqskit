"""This file tests the register_workflow function."""
from __future__ import annotations

from itertools import combinations
from random import choice

import pytest
from numpy import allclose

from bqskit.compiler import compile
from bqskit.compiler.machine import MachineModel
from bqskit.compiler.register import _workflow_registry
from bqskit.compiler.register import clear_registry
from bqskit.compiler.register import register_workflow
from bqskit.compiler.workflow import Workflow
from bqskit.compiler.workflow import WorkflowLike
from bqskit.ir import Circuit
from bqskit.ir import Gate
from bqskit.ir.gates import CZGate
from bqskit.ir.gates import HGate
from bqskit.ir.gates import RZGate
from bqskit.ir.gates import U3Gate
from bqskit.passes import QSearchSynthesisPass
from bqskit.passes import QuickPartitioner
from bqskit.passes import ScanningGateRemovalPass


def machine_match(mach_a: MachineModel, mach_b: MachineModel) -> bool:
    if mach_a.num_qudits != mach_b.num_qudits:
        return False
    if mach_a.radixes != mach_b.radixes:
        return False
    if mach_a.coupling_graph != mach_b.coupling_graph:
        return False
    if mach_a.gate_set != mach_b.gate_set:
        return False
    return True


def unitary_match(unit_a: Circuit, unit_b: Circuit) -> bool:
    return allclose(unit_a.get_unitary(), unit_b.get_unitary(), atol=1e-5)


def workflow_match(
    workflow_a: WorkflowLike,
    workflow_b: WorkflowLike,
) -> bool:
    if not isinstance(workflow_a, Workflow):
        workflow_a = Workflow(workflow_a)
    if not isinstance(workflow_b, Workflow):
        workflow_b = Workflow(workflow_b)
    if len(workflow_a) != len(workflow_b):
        return False
    for a, b in zip(workflow_a, workflow_b):
        if a.name != b.name:
            return False
    return True


def simple_circuit(num_qudits: int, gate_set: list[Gate]) -> Circuit:
    circ = Circuit(num_qudits)
    gate = choice(gate_set)
    if gate.num_qudits == 1:
        loc = choice(range(num_qudits))
    else:
        loc = choice(list(combinations(range(num_qudits), 2)))  # type: ignore
    gate_inv = gate.get_inverse()
    circ.append_gate(gate, loc)
    circ.append_gate(gate_inv, loc)
    return circ


class TestRegisterWorkflow:

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        # _workflow_registry.clear()
        clear_registry()

    def test_register_workflow(self) -> None:
        assert _workflow_registry == {}
        gateset = [CZGate(), HGate(), RZGate()]
        num_qudits = 3
        machine = MachineModel(num_qudits, gate_set=gateset)
        workflow = [QuickPartitioner(), ScanningGateRemovalPass()]
        register_workflow(machine, workflow)
        assert machine in _workflow_registry
        assert 1 in _workflow_registry[machine]
        assert workflow_match(_workflow_registry[machine][1], workflow)

    def test_custom_compile_machine(self) -> None:
        gateset = [CZGate(), HGate(), RZGate()]
        num_qudits = 3
        machine = MachineModel(num_qudits, gate_set=gateset)
        workflow = [QuickPartitioner(2)]
        register_workflow(machine, workflow)
        circuit = simple_circuit(num_qudits, gateset)
        result = compile(circuit, machine)
        assert unitary_match(result, circuit)
        assert result.num_operations > 0
        assert result.gate_counts != circuit.gate_counts
        result.unfold_all()
        assert result.gate_counts == circuit.gate_counts

    def test_custom_compile_gateset(self) -> None:
        gateset = [CZGate(), HGate(), RZGate()]
        num_qudits = 3
        machine = MachineModel(num_qudits, gate_set=gateset)
        workflow = [QuickPartitioner(2)]
        register_workflow(gateset, workflow)
        circuit = simple_circuit(num_qudits, gateset)
        result = compile(circuit, machine)
        assert unitary_match(result, circuit)
        assert result.num_operations > 0
        assert result.gate_counts != circuit.gate_counts
        result.unfold_all()
        assert result.gate_counts == circuit.gate_counts

    def test_custom_opt_level(self) -> None:
        gateset = [CZGate(), HGate(), RZGate()]
        num_qudits = 3
        machine = MachineModel(num_qudits, gate_set=gateset)
        workflow = [QSearchSynthesisPass()]
        register_workflow(gateset, workflow, 2)
        circuit = simple_circuit(num_qudits, gateset)
        result = compile(circuit, machine, optimization_level=2)
        assert unitary_match(result, circuit)
        assert result.gate_counts != circuit.gate_counts
        assert U3Gate() in result.gate_set
