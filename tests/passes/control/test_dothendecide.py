from __future__ import annotations

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.compiler import Compiler
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.passes.control import DoThenDecide


def accept_always(c1: Circuit, c2: Circuit) -> bool:
    return True


def reject_always(c1: Circuit, c2: Circuit) -> bool:
    return False


class AddCNOTPass(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        circuit.append_gate(CNOTGate(), (0, 1))


def test_dothendecide_accept(compiler: Compiler) -> None:
    circuit = Circuit(2)
    dtd_pass = DoThenDecide(accept_always, AddCNOTPass())
    out_circuit = compiler.compile(circuit, dtd_pass)
    assert out_circuit.num_operations == 1
    assert out_circuit.gate_set == {CNOTGate()}


def test_dothendecide_reject(compiler: Compiler) -> None:
    circuit = Circuit(2)
    dtd_pass = DoThenDecide(reject_always, AddCNOTPass())
    out_circuit = compiler.compile(circuit, dtd_pass)
    assert out_circuit.num_operations == 0


def test_dothendecide_list(compiler: Compiler) -> None:
    circuit = Circuit(2)
    dtd_pass = DoThenDecide(accept_always, [AddCNOTPass(), AddCNOTPass()])
    out_circuit = compiler.compile(circuit, dtd_pass)
    assert out_circuit.num_operations == 2
    assert out_circuit.gate_set == {CNOTGate()}
