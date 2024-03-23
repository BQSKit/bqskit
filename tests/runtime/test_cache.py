from __future__ import annotations

from numpy import float64
from numpy.random import randn
from numpy.typing import NDArray

from bqskit.compiler import BasePass
from bqskit.compiler import Compiler
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.passes.util.update import UpdateDataPass
from bqskit.runtime import get_runtime


def random_np_array() -> NDArray[float64]:
    return randn(4, 4, 4)


class TestWorkerCachePut(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        handle = get_runtime()
        cache = handle.get_cache()
        cache['random_array'] = data['random_array_put']


class TestWorkerCacheGet(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        handle = get_runtime()
        cache = handle.get_cache()
        data['random_array_get'] = cache['random_array']


def test_simple_put_and_get(server_compiler: Compiler) -> None:
    circuit = Circuit(2)
    passes = [
        UpdateDataPass('random_array_put', random_np_array()),
        TestWorkerCachePut(),
        TestWorkerCacheGet(),
    ]
    circuit, data = server_compiler.compile(circuit, passes, request_data=True)
    putted = data['random_array_put']
    gotten = data['random_array_get']
    assert (putted == gotten).all()
