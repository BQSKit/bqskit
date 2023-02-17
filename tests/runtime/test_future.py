from __future__ import annotations

import time
from typing import Any

from bqskit.compiler import BasePass
from bqskit.compiler import Compiler
from bqskit.ir import Circuit
from bqskit.runtime import get_runtime


def sleep1() -> None:
    time.sleep(1)


class TestPassFutureDone(BasePass):
    async def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        future = get_runtime().submit(sleep1)
        assert not future.done
        await future
        assert future.done


def test_future_done(compiler: Compiler) -> None:
    circuit = Circuit(2)
    compiler.compile(circuit, [TestPassFutureDone()])
