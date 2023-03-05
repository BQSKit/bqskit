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


class TestNextTask(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        future = get_runtime().map(sleepi, [3, 4, 1, 2])
        int_ids = await get_runtime().next(future)
        seen = [0]
        for int_id in int_ids:
            assert int_id[1] == max(seen) + 1
            seen.append(int_id[1])


class TestDoubleNextTask(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        future = get_runtime().map(sleepi, [3, 4, 1, 2])
        seen = [0]
        int_ids = await get_runtime().next(future)

        for int_id in int_ids:
            assert int_id[1] == max(seen) + 1
            seen.append(int_id[1])

        int_ids = await get_runtime().next(future)

        for int_id in int_ids:
            assert int_id[1] == max(seen) + 1
            seen.append(int_id[1])


class TestAwaitAfterNextTask(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        future = get_runtime().map(sleepi, [3, 4, 1, 2])
        _ = await get_runtime().next(future)
        assert (await future) == [3, 4, 1, 2]


class TestAwaitAfterDoubleNextTask(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        future = get_runtime().map(sleepi, [3, 4, 1, 2])
        _ = await get_runtime().next(future)
        _ = await get_runtime().next(future)
        assert (await future) == [3, 4, 1, 2]


class TestNextOnCompleteTask(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        future = get_runtime().map(sleepi, [3, 4, 1, 2])
        _ = await future
        with pytest.raises(RuntimeError):
            _ = await get_runtime().next(future)


class TestNextAfterSleepTask(BasePass):
    async def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        future = get_runtime().map(sleepi, [1, 2])
        time.sleep(1.5)
        int_ids = await get_runtime().next(future)
        assert len(int_ids) == 1  # Only one should have been received
        assert int_ids[0] == (0, 1)  # the 0th index 1 element


@pytest.mark.parametrize(
    'test_pass', [
        TestNextTask(),
        TestDoubleNextTask(),
        TestAwaitAfterNextTask(),
        TestAwaitAfterDoubleNextTask(),
        TestNextOnCompleteTask(),
        TestNextAfterSleepTask(),
    ],
)
def test_next_pass(server_compiler: Compiler, test_pass: BasePass) -> None:
    circuit = Circuit(2)
    server_compiler.compile(circuit, [test_pass])
