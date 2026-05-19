from __future__ import annotations

from bqskit.compiler import BasePass
from bqskit.compiler import PassData
from bqskit.compiler import Workflow
from bqskit.ir import Circuit
from bqskit.utils.citation import Citation
from bqskit.utils.citation import cite


def test_citations() -> None:
    @cite(key='test1', bibtex='@article{test1}')
    class PassA(BasePass):
        async def run(self, circuit: Circuit, data: PassData) -> None:
            pass

    @cite(key='test2', bibtex='@article{test2}')
    class PassB(BasePass):
        async def run(self, circuit: Circuit, data: PassData) -> None:
            pass

    @cite(key='test1', bibtex='@article{test1}')  # duplicate
    class PassC(BasePass):
        async def run(self, circuit: Circuit, data: PassData) -> None:
            pass

    a, b, c = PassA(), PassB(), PassC()

    assert a.get_citations() == {
        Citation(
            key='test1', bibtex='@article{test1}',
        ),
    }
    assert b.get_citations() == {
        Citation(
            key='test2', bibtex='@article{test2}',
        ),
    }

    workflow = Workflow([a, b, c])
    gathered = workflow.gather_citations()

    assert len(gathered) == 2  # test1 and test2, deduplicated
    assert set(gathered.keys()) == {
        Citation(key='test1', bibtex='@article{test1}'),
        Citation(key='test2', bibtex='@article{test2}'),
    }

    cite1 = Citation(key='test1', bibtex='@article{test1}')
    assert set(gathered[cite1]) == {a, c}  # both passes with test1
