"""Checks the runtime's ability and correctness when executing tasks."""
from __future__ import annotations

from typing import Any

from bqskit.compiler import BasePass
from bqskit.compiler import Compiler
from bqskit.ir import Circuit
from bqskit.ir.gates import HGate
from bqskit.ir.gates import XGate
from bqskit.runtime import get_runtime


def iden(i: int) -> int:
    return i


async def parent(i: int) -> tuple[int, int]:
    return await get_runtime().map(iden, [2 * i, 2 * i + 1])


async def add_hgate_to_circuit(c: Circuit) -> Circuit:
    c.append_gate(HGate(), 0)
    return c


async def add_xgate_to_circuit(c: Circuit) -> Circuit:
    c.append_gate(XGate(), 0)
    return c


class TestPass1(BasePass):
    async def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        results = await get_runtime().map(iden, list(range(10)))
        assert results == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


class TestPass2(BasePass):
    async def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        results = await get_runtime().map(parent, list(range(1000)))
        assert len(results) == 1000
        assert all(r == [2 * i, 2 * i + 1] for i, r in enumerate(results))


class TestPass3(BasePass):
    async def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        out = await get_runtime().submit(add_hgate_to_circuit, circuit)
        circuit.become(out)


class TestPass4(BasePass):
    async def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        out = await get_runtime().submit(add_xgate_to_circuit, circuit)
        circuit.become(out)


def test_simple_map(server_compiler: Compiler) -> None:
    circuit = Circuit(2)
    server_compiler.compile(circuit, [TestPass1()])


def test_2level_map(server_compiler: Compiler) -> None:
    circuit = Circuit(2)
    server_compiler.compile(circuit, [TestPass2()])


def test_circuit_change(server_compiler: Compiler) -> None:
    circuit = Circuit(2)
    out_circuit = server_compiler.compile(circuit, [TestPass3()])
    assert HGate() in out_circuit.gate_set
    assert out_circuit.num_operations == 1


def test_reuse(server_compiler: Compiler) -> None:
    circuit = Circuit(2)
    server_compiler.compile(circuit, [TestPass1()])
    server_compiler.compile(circuit, [TestPass1()])


def test_parallel(server_compiler: Compiler) -> None:
    circuit = Circuit(2)
    t1 = server_compiler.submit(circuit, [TestPass3()])
    t2 = server_compiler.submit(circuit, [TestPass4()])
    c1 = server_compiler.result(t1)
    c2 = server_compiler.result(t2)
    assert isinstance(c1, Circuit)
    assert isinstance(c2, Circuit)
    assert HGate() in c1.gate_set
    assert c1.num_operations == 1
    assert XGate() in c2.gate_set
    assert c2.num_operations == 1
