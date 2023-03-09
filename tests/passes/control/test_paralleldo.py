from __future__ import annotations

import time

import pytest

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.compiler import Compiler
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import XGate
from bqskit.ir.gates import YGate
from bqskit.ir.gates import ZGate
from bqskit.passes.control.paralleldo import ParallelDo


class AddXGate(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        circuit.append_gate(XGate(), 0)
        data['key'] = 'x'


class AddYGate(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        circuit.append_gate(YGate(), 0)
        data['key'] = 'y'


class AddZGate(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        circuit.append_gate(ZGate(), 0)
        data['key'] = 'z'


class Sleep1Pass(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        time.sleep(0.1)
        data['key'] = '1'


class Sleep3Pass(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        circuit.append_gate(ZGate(), 0)
        time.sleep(0.3)
        data['key'] = '3'


def pick_z(c1: Circuit, c2: Circuit) -> bool:
    if ZGate() in c1.gate_set:
        return True
    else:
        return False


def test_parallel_do(compiler: Compiler) -> None:
    pd_pass = ParallelDo([[AddXGate()], [AddYGate()], [AddZGate()]], pick_z)
    out_circuit, data = compiler.compile(Circuit(1), pd_pass, True)
    assert ZGate() in out_circuit.gate_set
    assert len(out_circuit.gate_set) == 1
    assert data['key'] == 'z'


def test_parallel_do_no_passes() -> None:
    with pytest.raises(ValueError):
        ParallelDo([], pick_z)


def test_parallel_do_pick_first(compiler: Compiler) -> None:
    passes: list[list[BasePass]] = [[Sleep3Pass()], [Sleep1Pass()]]
    pd_pass = ParallelDo(passes, pick_z, True)
    _, data = compiler.compile(Circuit(1), pd_pass, True)
    assert data['key'] == '1'
