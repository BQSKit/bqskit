from __future__ import annotations

import time
from typing import Any

import pytest

from bqskit.compiler import BasePass
from bqskit.compiler import Compiler
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.runtime import get_runtime


def iden(i: Any) -> Any:
    return i


def sleep1() -> None:
    time.sleep(1)


class TestCancelTask(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        future = get_runtime().submit(sleep1)
        get_runtime().cancel(future)


class TestCantAwaitCancelTask(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        future = get_runtime().submit(sleep1)
        get_runtime().cancel(future)
        await future


def test_simple_cancel(server_compiler: Compiler) -> None:
    circuit = Circuit(2)
    server_compiler.compile(circuit, [TestCancelTask()])


def test_cant_await_cancel(server_compiler: Compiler) -> None:
    circuit = Circuit(2)
    with pytest.raises(RuntimeError):
        server_compiler.compile(circuit, [TestCantAwaitCancelTask()])
