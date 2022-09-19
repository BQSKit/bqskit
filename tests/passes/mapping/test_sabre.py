from __future__ import annotations

from bqskit.compiler import CompilationTask
from bqskit.compiler import Compiler
from bqskit.compiler import MachineModel
from bqskit.ir import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import U3Gate
from bqskit.passes import GeneralizedSabreLayoutPass
from bqskit.passes import GeneralizedSabreRoutingPass
from bqskit.passes import GreedyPlacementPass
from bqskit.passes import SetModelPass
from bqskit.qis import PermutationMatrix
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


def test_simple(compiler: Compiler) -> None:
    model = MachineModel(
        8, [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)],
    )
    circuit = Circuit(5)
    for i in range(4):
        circuit.append_gate(CNOTGate(), (4, i))
        circuit.append_gate(
            U3Gate(), 4, U3Gate.calc_params(
                UnitaryMatrix.random(1),
            ),
        )
        circuit.append_gate(
            U3Gate(), i, U3Gate.calc_params(
                UnitaryMatrix.random(1),
            ),
        )
    circuit.append_gate(CNOTGate(), (0, 1))

    in_utry = circuit.get_unitary()

    task = CompilationTask(
        circuit,
        [
            SetModelPass(model),
            GreedyPlacementPass(),
            GeneralizedSabreLayoutPass(),
            GeneralizedSabreRoutingPass(),
        ],
    )

    cc = compiler.compile(task)
    pi = compiler.analyze(task, 'initial_mapping')
    pf = compiler.analyze(task, 'final_mapping')
    PI = PermutationMatrix.from_qubit_location(5, pi)
    PF = PermutationMatrix.from_qubit_location(5, pf)
    assert cc.get_unitary().get_distance_from(PF.T @ in_utry @ PI) < 1e-7
