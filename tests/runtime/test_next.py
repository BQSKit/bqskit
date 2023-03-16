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


class TestNoDuplicateResult(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        future = get_runtime().map(sleepi, [0.3, 0.4, 0.1, 0.2])
        int_ids = await get_runtime().next(future)
        seen = [0]
        for int_id in int_ids:
            assert not int_id[1] in seen
            seen.append(int_id[1])


class TestNoDuplicateResultsInTwoNexts(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        future = get_runtime().map(sleepi, [0.3, 0.4, 0.1, 0.2])
        seen = [0]
        int_ids = await get_runtime().next(future)

        for int_id in int_ids:
            assert not int_id[1] in seen
            seen.append(int_id[1])

        int_ids = await get_runtime().next(future)

        for int_id in int_ids:
            assert not int_id[1] in seen
            seen.append(int_id[1])


class TestFutureAwaitGivesAllResultsAfterNext(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        sleep_times = [0.3, 0.4, 0.1, 0.2]
        future = get_runtime().map(sleepi, sleep_times)
        _ = await get_runtime().next(future)
        assert (await future) == sleep_times


class TestNextOnCompleteTaskRaisesError(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        future = get_runtime().map(sleepi, [0.3, 0.4, 0.1, 0.2])
        _ = await future
        with pytest.raises(RuntimeError):
            _ = await get_runtime().next(future)


class TestNextNeverlosesResults(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        future = get_runtime().map(sleepi, [0.1])
        time.sleep(0.2)
        int_ids = await get_runtime().next(future)
        assert len(int_ids) == 1  # Only one should have been received
        assert int_ids[0] == (0, .1)  # the 0th index 1 element


@pytest.mark.parametrize(
    'test_pass', [
        TestNoDuplicateResult(),
        TestNoDuplicateResultsInTwoNexts(),
        TestFutureAwaitGivesAllResultsAfterNext(),
        TestNextOnCompleteTaskRaisesError(),
        TestNextNeverlosesResults(),
    ],
)
def test_next_pass(server_compiler: Compiler, test_pass: BasePass) -> None:
    circuit = Circuit(2)
    server_compiler.compile(circuit, [test_pass])
