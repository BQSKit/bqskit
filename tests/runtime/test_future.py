from __future__ import annotations

import time
from typing import Any

import pytest

from bqskit.compiler import BasePass
from bqskit.compiler import Compiler
from bqskit.ir import Circuit
from bqskit.runtime import get_runtime


def iden(i: Any) -> Any:
    return i


def sleep1() -> None:
    time.sleep(1)


class TestPassFutureDone(BasePass):
    async def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        future = get_runtime().submit(sleep1)
        assert not future.done
        await future
        assert future.done


class TestPassFutureCannotSend(BasePass):
    async def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        future = get_runtime().submit(sleep1)
        get_runtime().submit(iden, future)


def test_future_done(compiler: Compiler) -> None:
    circuit = Circuit(2)
    compiler.compile(circuit, [TestPassFutureDone()])


def test_future_cannot_send(fresh_compiler: Compiler) -> None:
    circuit = Circuit(2)
    with pytest.raises(RuntimeError):
        fresh_compiler.compile(circuit, [TestPassFutureCannotSend()])
