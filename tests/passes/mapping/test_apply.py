from __future__ import annotations

from typing import Any

from bqskit.compiler import MachineModel
from bqskit.ir import Circuit
from bqskit.passes import ApplyPlacement
from bqskit.passes import GreedyPlacementPass
from bqskit.passes import SetModelPass


def test_apply_placement(r3_qubit_circuit: Circuit) -> None:
    model = MachineModel(6, [(0, 1), (1, 2), (2, 3), (2, 4), (2, 5), (3, 5)])
    data: dict[str, Any] = {}
    SetModelPass(model).run(r3_qubit_circuit, data)
    GreedyPlacementPass().run(r3_qubit_circuit, data)
    print(data)
    ApplyPlacement().run(r3_qubit_circuit, data)
    assert r3_qubit_circuit.num_qudits == 6
    assert r3_qubit_circuit.active_qudits == [2, 3, 5]
