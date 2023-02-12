from __future__ import annotations

from bqskit.compiler import Compiler
from bqskit.compiler import MachineModel
from bqskit.compiler.task import CompilationTask
from bqskit.ir import Circuit
from bqskit.passes import ApplyPlacement
from bqskit.passes import GreedyPlacementPass
from bqskit.passes import SetModelPass


def test_apply_placement(r3_qubit_circuit: Circuit) -> None:
    model = MachineModel(6, [(0, 1), (1, 2), (2, 3), (2, 4), (2, 5), (3, 5)])
    workflow = [SetModelPass(model), GreedyPlacementPass(), ApplyPlacement()]
    task = CompilationTask(r3_qubit_circuit, workflow)
    with Compiler() as compiler:
        out_circuit = compiler.compile(task)
        assert out_circuit.num_qudits == 6
        assert out_circuit.active_qudits == [2, 3, 5]
