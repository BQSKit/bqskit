"""This module tests the runtime's ability to wake tasks on first result."""
from __future__ import annotations

import time

import pytest

from bqskit.compiler import BasePass
from bqskit.compiler import Compiler
from bqskit.compiler.passdata import PassData
from bqskit.ir import Circuit
from bqskit.runtime import get_runtime


def sleepi(i: int) -> int:
    time.sleep(i)
    return i


class TestWaitTask(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        future = get_runtime().map(sleepi, [3, 4, 1, 2])
        int_ids = await get_runtime().wait(future)
        assert len(int_ids) == 1  # The 2, 3, and 4 sec sleep shouldn't arrive
        assert int_ids[0] == (2, 1)  # 2nd index in map and result is 1


class TestDoubleWaitTask(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        future = get_runtime().map(sleepi, [3, 4, 1, 2])
        _ = await get_runtime().wait(future)
        int_ids = await get_runtime().wait(future)
        assert len(int_ids) == 1  # The 3 and 4 sec sleep shouldn't arrive
        assert int_ids[0] == (3, 2)  # 2nd index in map and result is 1


class TestAwaitAfterWaitTask(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        future = get_runtime().map(sleepi, [3, 4, 1, 2])
        _ = await get_runtime().wait(future)
        assert (await future) == [3, 4, 1, 2]


class TestAwaitAfterDoubleWaitTask(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        future = get_runtime().map(sleepi, [3, 4, 1, 2])
        _ = await get_runtime().wait(future)
        _ = await get_runtime().wait(future)
        assert (await future) == [3, 4, 1, 2]


class TestWaitOnCompleteTask(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        future = get_runtime().map(sleepi, [3, 4, 1, 2])
        _ = await future
        with pytest.raises(RuntimeError):
            _ = await get_runtime().wait(future)


@pytest.mark.parametrize(
    'test_pass', [
        TestWaitTask(),
        TestDoubleWaitTask(),
        TestAwaitAfterWaitTask(),
        TestAwaitAfterDoubleWaitTask(),
        TestWaitOnCompleteTask(),
    ],
)
def test_wait_pass(server_compiler: Compiler, test_pass: BasePass) -> None:
    circuit = Circuit(2)
    server_compiler.compile(circuit, [test_pass])
