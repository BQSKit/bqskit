from __future__ import annotations

from bqskit.compiler import BasePass
from bqskit.compiler import PassData
from bqskit.compiler import Workflow
from bqskit.ir import Circuit
from bqskit.utils.citation import cite


def test_citations() -> None:
    @cite(doi='test1')
    class PassA(BasePass):
        async def run(self, circuit: Circuit, data: PassData) -> None:
            pass

    @cite(doi='test2')
    class PassB(BasePass):
        async def run(self, circuit: Circuit, data: PassData) -> None:
            pass

    @cite(doi='test1')  # duplicate
    class PassC(BasePass):
        async def run(self, circuit: Circuit, data: PassData) -> None:
            pass

    a, b, c = PassA(), PassB(), PassC()

    assert a.get_citations() == {'test1'}
    assert b.get_citations() == {'test2'}

    workflow = Workflow([a, b, c])
    gathered = workflow.gather_citations()

    assert len(gathered) == 2  # test1 and test2, deduplicated
    assert set(gathered['test1']) == {
        'PassA',
        'PassC',
    }  # both passes with test1
