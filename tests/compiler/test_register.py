"""This file tests the register_workflow function."""
from __future__ import annotations

from itertools import combinations
from random import choice

from bqskit.compiler import compile
from bqskit.compiler.machine import MachineModel
from bqskit.compiler.register import register_workflow
from bqskit.compiler.register import workflow_registry
from bqskit.compiler.workflow import Workflow
from bqskit.compiler.workflow import WorkflowLike
from bqskit.ir import Circuit
from bqskit.ir import Gate
from bqskit.ir.gates import CZGate
from bqskit.ir.gates import HGate
from bqskit.ir.gates import RZGate
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

    def test_register_workflow(self) -> None:
        assert workflow_registry == {}
        machine = MachineModel(3)
        workflow = [QuickPartitioner(), ScanningGateRemovalPass()]
        register_workflow(machine, workflow)
        assert machine in workflow_registry
        assert 1 in workflow_registry[machine]
        assert workflow_match(workflow_registry[machine][1], workflow)

    def test_custom_compile_machine(self) -> None:
        gateset = [CZGate(), HGate(), RZGate()]
        num_qudits = 3
        machine = MachineModel(num_qudits, gate_set=gateset)
        workflow = [QuickPartitioner(2)]
        register_workflow(machine, workflow)
        circuit = simple_circuit(num_qudits, gateset)
        result = compile(circuit, machine)
        assert result.get_unitary() == circuit.get_unitary()
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
        assert result.get_unitary() == circuit.get_unitary()
        assert result.num_operations > 0
        assert result.gate_counts != circuit.gate_counts
        result.unfold_all()
        assert result.gate_counts == circuit.gate_counts
