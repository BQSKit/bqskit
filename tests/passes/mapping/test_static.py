from __future__ import annotations

import pytest

from bqskit import Circuit
from bqskit import MachineModel
from bqskit.ir.gates import CNOTGate
from bqskit.qis import CouplingGraph

from bqskit.compiler import Compiler
from bqskit.compiler import MachineModel
from bqskit.ir.circuit import Circuit
from bqskit.passes import IfThenElsePass, LogPass
from bqskit.passes import SetModelPass
from bqskit.passes import ApplyPlacement
from bqskit.passes import GreedyPlacementPass
from bqskit.passes import StaticPlacementPass
from bqskit.passes.control.predicates import PhysicalPredicate


def circular_circuit(n: int) -> Circuit:
    circuit = Circuit(n)
    for i in range(n):
        circuit.append_gate(CNOTGate(), [i, (i + 1) % n])
    return circuit


@pytest.mark.parametrize(
    ["grid_size", "logical_qudits"],
    sum([[(n, i) for i in range(2, n**2, 2)] for n in range(2, 8)], [])
    + sum([[(n, i) for i in range(3, n**2, 2)] for n in range(2, 6)], []),
)
def test_circular_to_grid(
    grid_size: int, logical_qudits: int, compiler: Compiler
) -> None:
    circuit = circular_circuit(logical_qudits)
    cg = CouplingGraph.grid(grid_size, grid_size)
    model = MachineModel(grid_size**2, cg)
    workflow = [
        SetModelPass(model),
        StaticPlacementPass(timeout_sec=1.0),
        IfThenElsePass(
            PhysicalPredicate(),
            [LogPass("Static Placement Found")],
            [LogPass("Greedy Placement Required"), GreedyPlacementPass()],
        ),
        ApplyPlacement(),
    ]
    out_circuit = compiler.compile(circuit, workflow)
