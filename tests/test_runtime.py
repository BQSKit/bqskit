import time
from typing import Any
from bqskit.compiler import Compiler, BasePass
from bqskit.compiler.task import CompilationTask
from bqskit.ir.circuit import Circuit

def test_startup_transparently() -> None:
    compiler = Compiler(num_workers=1)
    assert compiler.p is not None

def test_double_close() -> None:
    compiler = Compiler(num_workers=1)
    assert compiler.p is not None
    compiler.close()
    assert compiler.p is None
    compiler.close()
    assert compiler.p is None

from bqskit.runtime import get_runtime

def iden(i: int) -> None:
    return i

async def parent(i: int) -> tuple[int, int]:
    return await get_runtime().map(iden, [2*i, 2*i + 1])

class TestPass1(BasePass):
    async def run(self, circuit, data = {}):
        results = await get_runtime().map(iden, list(range(10)))
        assert results == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

class TestPass2(BasePass):
    async def run(self, circuit, data = {}):
        results = await get_runtime().map(parent, list(range(1000)))
        assert len(results) == 1000
        assert all(r == [2*i, 2*i + 1] for i, r in enumerate(results))

def test_simple_map() -> None:
    circuit = Circuit(2)
    compiler = Compiler()
    task = CompilationTask(circuit, [TestPass1()])
    compiler.compile(task)
    compiler.close()

def test_2level_map() -> None:
    circuit = Circuit(2)
    compiler = Compiler()
    task = CompilationTask(circuit, [TestPass2()])
    compiler.compile(task)
    compiler.close()


def sleep1() -> None:
    time.sleep(1)

class TestPassFutureDone(BasePass):
    async def run(self, circuit, data = {}):
        future = get_runtime().submit(sleep1)
        assert not future.done
        await future
        assert future.done

def test_future_done() -> None:
    circuit = Circuit(2)
    compiler = Compiler()
    task = CompilationTask(circuit, [TestPassFutureDone()])
    compiler.compile(task)
    compiler.close()
