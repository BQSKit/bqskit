"""This file tests the register_workflow function."""
from __future__ import annotations

from itertools import combinations
from random import choice

import pytest
from numpy import allclose

from bqskit.compiler.compile import compile
from bqskit.compiler.machine import MachineModel
from bqskit.compiler.registry import _compile_circuit_registry
from bqskit.compiler.registry import _compile_statemap_registry
from bqskit.compiler.registry import _compile_stateprep_registry
from bqskit.compiler.registry import _compile_unitary_registry
from bqskit.compiler.registry import register_workflow
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
        # global _compile_registry
        _compile_circuit_registry.clear()
        _compile_unitary_registry.clear()
        _compile_statemap_registry.clear()
        _compile_stateprep_registry.clear()

    def test_register_workflow(self) -> None:
        assert _compile_circuit_registry == {}
        assert _compile_unitary_registry == {}
        assert _compile_statemap_registry == {}
        assert _compile_stateprep_registry == {}
        gateset = [CZGate(), HGate(), RZGate()]
        num_qudits = 3
        machine = MachineModel(num_qudits, gate_set=gateset)
        circuit_workflow = [QuickPartitioner(), ScanningGateRemovalPass()]
        other_workflow = [QuickPartitioner(), QSearchSynthesisPass()]
        register_workflow(machine, circuit_workflow, 1, 'circuit')
        register_workflow(machine, other_workflow, 1, 'unitary')
        register_workflow(machine, other_workflow, 1, 'statemap')
        register_workflow(machine, other_workflow, 1, 'stateprep')
        assert machine in _compile_circuit_registry
        assert 1 in _compile_circuit_registry[machine]
        assert workflow_match(
            _compile_circuit_registry[machine][1], circuit_workflow,
        )
        assert machine in _compile_unitary_registry
        assert 1 in _compile_unitary_registry[machine]
        assert workflow_match(
            _compile_unitary_registry[machine][1], other_workflow,
        )
        assert machine in _compile_statemap_registry
        assert 1 in _compile_statemap_registry[machine]
        assert workflow_match(
            _compile_statemap_registry[machine][1], other_workflow,
        )
        assert machine in _compile_stateprep_registry
        assert 1 in _compile_stateprep_registry[machine]
        assert workflow_match(
            _compile_stateprep_registry[machine][1], other_workflow,
        )

    def test_custom_compile_machine(self) -> None:
        assert _compile_circuit_registry == {}
        gateset = [CZGate(), HGate(), RZGate()]
        num_qudits = 3
        machine = MachineModel(num_qudits, gate_set=gateset)
        workflow = [QuickPartitioner(2)]
        register_workflow(machine, workflow, 1, 'circuit')
        circuit = simple_circuit(num_qudits, gateset)
        result = compile(circuit, machine)
        assert unitary_match(result, circuit)
        assert result.num_operations > 0
        assert result.gate_counts != circuit.gate_counts
        result.unfold_all()
        assert result.gate_counts == circuit.gate_counts

    def test_custom_opt_level(self) -> None:
        assert _compile_circuit_registry == {}
        gateset = [CZGate(), HGate(), RZGate()]
        num_qudits = 3
        machine = MachineModel(num_qudits, gate_set=gateset)
        workflow = [QSearchSynthesisPass()]
        register_workflow(machine, workflow, 2, 'circuit')
        circuit = simple_circuit(num_qudits, gateset)
        result = compile(circuit, machine, optimization_level=2)
        assert unitary_match(result, circuit)
        assert result.gate_counts != circuit.gate_counts
        assert U3Gate() in result.gate_set
