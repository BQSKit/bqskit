import pytest
import os
from typing import Any
from bqskit.passes import QuickPartitioner
from bqskit.passes import UnfoldPass
from bqskit import Circuit


@pytest.fixture(
    params=os.listdir(os.path.join(__file__[:__file__.rfind('/')], '_data')),
    ids=lambda qasm_file: qasm_file[qasm_file.rfind('/')+1:-5]
)
def big_qasm_files(request: Any) -> Circuit:
    """Provide location of a big qasm file."""
    cur_dir = __file__[:__file__.rfind('/')]
    path = os.path.join(cur_dir, '_data')
    return os.path.join(path, request.param)


def test_parters(big_qasm_file: str) -> None:
    c = Circuit.from_file(big_qasm_file)
    wc = c.copy()
    QuickPartitioner(3).run(wc)
    UnfoldPass().run(wc)

    for q in range(c.num_qudits):
        gate_list1 = list(c.operations_with_cycles(qudits_or_region=[q]))
        gate_list2 = list(wc.operations_with_cycles(qudits_or_region=[q]))

        # Each qudit should have same number of gates on them
        assert len(gate_list1) == len(gate_list2)

        for g1, g2 in zip(gate_list1, gate_list2):
            # Order of operations on one qudit should be same
            assert g1[1] == g2[1]

            # Each operation should have the same previous ops
            prevs1 = c.prev((g1[0], g1[1].location[0]))
            prevs2 = wc.prev((g2[0], g2[1].location[0]))

            prev_ops1 = [c[prev] for prev in prevs1]
            prev_ops2 = [c[prev] for prev in prevs2]

            assert all(o in prev_ops2 for o in prev_ops1)
            assert all(o in prev_ops1 for o in prev_ops2)

            # Each operation should have the same next ops
            nexts1 = c.next((g1[0], g1[1].location[0]))
            nexts2 = wc.next((g2[0], g2[1].location[0]))

            next_ops1 = [c[next] for next in nexts1]
            next_ops2 = [c[next] for next in nexts2]

            assert all(o in next_ops2 for o in next_ops1)
            assert all(o in next_ops1 for o in next_ops2)
