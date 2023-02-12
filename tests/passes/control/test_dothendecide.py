from __future__ import annotations

from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.passes.control import DoThenDecide


def accept_always(c1: Circuit, c2: Circuit) -> bool:
    return True


def reject_always(c1: Circuit, c2: Circuit) -> bool:
    return False


class AddCNOTPass(BasePass):
    async def run(
        self,
        circuit: Circuit,
        data: dict[str, Any] = {},
    ) -> None:
        circuit.append_gate(CNOTGate(), (0, 1))


def test_dothendecide_accept() -> None:
    circuit = Circuit(2)
    dtd_pass = DoThenDecide(accept_always, AddCNOTPass())
    circuit.perform(dtd_pass)
    assert circuit.num_operations == 1
    assert circuit.gate_set == {CNOTGate()}


def test_dothendecide_reject() -> None:
    circuit = Circuit(2)
    dtd_pass = DoThenDecide(reject_always, AddCNOTPass())
    circuit.perform(dtd_pass)
    assert circuit.num_operations == 0


def test_dothendecide_list() -> None:
    circuit = Circuit(2)
    dtd_pass = DoThenDecide(accept_always, [AddCNOTPass(), AddCNOTPass()])
    circuit.perform(dtd_pass)
    assert circuit.num_operations == 2
    assert circuit.gate_set == {CNOTGate()}
