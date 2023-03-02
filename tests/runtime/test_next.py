"""This module tests the runtime's ability to wake tasks on first result."""
from __future__ import annotations

import time
from typing import Any

import pytest

from bqskit.compiler import BasePass
from bqskit.compiler import Compiler
from bqskit.ir import Circuit
from bqskit.runtime import get_runtime


def sleepi(i: int) -> int:
    time.sleep(i)
    return i


class TestNextTask(BasePass):
    async def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        future = get_runtime().map(sleepi, [3, 4, 1, 2])
        int_ids = await get_runtime().next(future)
        assert len(int_ids) == 1  # The 2, 3, and 4 sec sleep shouldn't arrive
        assert int_ids[0] == (2, 1)  # 2nd index in map and result is 1


class TestDoubleNextTask(BasePass):
    async def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        future = get_runtime().map(sleepi, [3, 4, 1, 2])
        _ = await get_runtime().next(future)
        int_ids = await get_runtime().next(future)
        assert len(int_ids) == 1  # The 3 and 4 sec sleep shouldn't arrive
        assert int_ids[0] == (3, 2)  # 2nd index in map and result is 1


class TestAwaitAfterNextTask(BasePass):
    async def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        future = get_runtime().map(sleepi, [3, 4, 1, 2])
        _ = await get_runtime().next(future)
        assert (await future) == [3, 4, 1, 2]


class TestAwaitAfterDoubleNextTask(BasePass):
    async def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        future = get_runtime().map(sleepi, [3, 4, 1, 2])
        _ = await get_runtime().next(future)
        _ = await get_runtime().next(future)
        assert (await future) == [3, 4, 1, 2]


class TestNextOnCompleteTask(BasePass):
    async def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        future = get_runtime().map(sleepi, [3, 4, 1, 2])
        _ = await future
        with pytest.raises(RuntimeError):
            _ = await get_runtime().next(future)


@pytest.mark.parametrize(
    'test_pass', [
        TestNextTask(),
        TestDoubleNextTask(),
        TestAwaitAfterNextTask(),
        TestAwaitAfterDoubleNextTask(),
        TestNextOnCompleteTask(),
    ],
)
def test_next_pass(server_compiler: Compiler, test_pass: BasePass) -> None:
    circuit = Circuit(2)
    server_compiler.compile(circuit, [test_pass])
