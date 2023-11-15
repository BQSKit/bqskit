from __future__ import annotations

import pytest

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.compiler import Compiler
from bqskit.compiler.passdata import PassData
from bqskit.compiler.workflow import Workflow
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CircuitGate
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import HGate
from bqskit.ir.gates import XGate
from bqskit.ir.gates import YGate
from bqskit.ir.gates import ZGate
from bqskit.ir.operation import Operation
from bqskit.passes import UnfoldPass
from bqskit.passes.control.foreach import ForEachBlockPass
from bqskit.passes.partitioning.quick import QuickPartitioner
from bqskit.passes.util.update import UpdateDataPass


@pytest.fixture
def blk_circuit() -> Circuit:
    circuit = Circuit(3)

    for gate, loc in [(XGate(), (0, 1)), (YGate(), (1, 2)), (ZGate(), (0, 1))]:
        block = Circuit(2)
        block.append_gate(gate, 0)
        block.append_gate(gate, 1)
        block_gate = CircuitGate(block)

        circuit.append_gate(block_gate, loc)

    return circuit


class RecordGateSetPass(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        data['gate_set'] = circuit.gate_set


class ReplaceXwithHPass(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        if XGate() in circuit.gate_set:
            circuit.pop()
            circuit.pop()
            circuit.append_gate(HGate(), 0)
            circuit.append_gate(HGate(), 1)
            circuit.append_gate(HGate(), 0)
            circuit.append_gate(HGate(), 1)


class TestForEachPassDownSpecificSeeds(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        key = ForEachBlockPass.pass_down_block_specific_key_prefix
        key += 'test_info'
        assert key in data
        assert data[key] == 'a' or data[key] == 'b'
        if data[key] == 'a':
            data['test_response'] = 'a'
        else:
            data['test_response'] = 'b'


def never_replace(c: Circuit, o: Operation) -> bool:
    return False


class RemoveXGatePass(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        if XGate() in circuit.gate_set:
            circuit.pop()
            circuit.pop()


def only_x(o: Operation) -> bool:
    return o.gate == XGate()


class AssertXGatePass(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        assert XGate() in circuit.gate_set
        assert len(circuit.gate_set) == 1


def empty_coll(_: Operation) -> bool:
    return False


def test_every_block_touched(blk_circuit: Circuit, compiler: Compiler) -> None:
    feb_pass = ForEachBlockPass(RecordGateSetPass())
    _, out_data = compiler.compile(blk_circuit, feb_pass, True)
    feb_data = out_data[ForEachBlockPass.key]
    assert len(feb_data) == 1
    assert len(feb_data[0]) == 3
    touched_gates = set()
    for x in feb_data[0]:
        assert len(x['gate_set']) == 1
        touched_gates.update(x['gate_set'])
    assert len(touched_gates) == 3
    assert all(g in touched_gates for g in [XGate(), YGate(), ZGate()])


def test_replacement_happens(blk_circuit: Circuit, compiler: Compiler) -> None:
    feb_pass = ForEachBlockPass(ReplaceXwithHPass())
    out_circuit = compiler.compile(blk_circuit, [feb_pass, UnfoldPass()])
    assert len(out_circuit) == 8
    assert len(out_circuit.gate_set) == 3
    assert all(g in out_circuit.gate_set for g in [HGate(), YGate(), ZGate()])


def test_replace_filter(blk_circuit: Circuit, compiler: Compiler) -> None:
    feb_pass = ForEachBlockPass(
        ReplaceXwithHPass(), replace_filter=never_replace,
    )
    out_circuit = compiler.compile(blk_circuit, [feb_pass, UnfoldPass()])
    assert len(out_circuit) == 6
    assert len(out_circuit.gate_set) == 3
    assert all(g in out_circuit.gate_set for g in [XGate(), YGate(), ZGate()])


def test_calculate_error(blk_circuit: Circuit, compiler: Compiler) -> None:
    in_utry = blk_circuit.get_unitary()
    feb_pass = ForEachBlockPass(RemoveXGatePass(), calculate_error_bound=True)
    workflow = [feb_pass, UnfoldPass()]
    out_circuit, data = compiler.compile(blk_circuit, workflow, True)
    out_utry = out_circuit.get_unitary()
    dist = out_utry.get_distance_from(in_utry)
    assert data.error == dist


def test_collection_filter(blk_circuit: Circuit, compiler: Compiler) -> None:
    blk_circuit.unfold_all()
    feb_pass = ForEachBlockPass(AssertXGatePass(), collection_filter=only_x)
    compiler.compile(blk_circuit, feb_pass)


def test_no_hang_on_empty_circuit(compiler: Compiler) -> None:
    feb_pass = ForEachBlockPass(RemoveXGatePass(), collection_filter=empty_coll)
    compiler.compile(Circuit(1), feb_pass)


def test_no_hang_on_empty_collection(compiler: Compiler) -> None:
    circuit = Circuit(1)
    circuit.append_gate(XGate(), 0)
    feb_pass = ForEachBlockPass(RemoveXGatePass(), collection_filter=empty_coll)
    compiler.compile(circuit, feb_pass)


def test_pass_down_seeds(compiler: Compiler) -> None:
    circuit = Circuit(4)
    circuit.append_gate(CNOTGate(), (0, 2))
    circuit.append_gate(CNOTGate(), (1, 2))
    circuit.append_gate(CNOTGate(), (1, 3))
    circuit.append_gate(CNOTGate(), (2, 3))

    # Manually set seed for blocks 0 and 1
    input_info = {0: 'a', 1: 'b'}

    partitioner = QuickPartitioner()
    check_specific = TestForEachPassDownSpecificSeeds()
    foreach = ForEachBlockPass(check_specific)

    key = foreach.pass_down_block_specific_key_prefix + 'test_info'
    updater = UpdateDataPass(key, input_info)

    # For checking specific pass down data exists
    workflow = Workflow([partitioner, updater, foreach])

    # Check usability of block specific keys
    compiled, data = compiler.compile(
        circuit, workflow, request_data=True,
    )

    block0_data = data['ForEachBlockPass_data'][0][0]
    block1_data = data['ForEachBlockPass_data'][0][1]
    response_key = 'test_response'
    assert response_key in block0_data
    assert response_key in block1_data
    assert block0_data[response_key] == 'a'
    assert block1_data[response_key] == 'b'
