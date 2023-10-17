from __future__ import annotations

from bqskit.compiler import Compiler
from bqskit.compiler import MachineModel
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import U3Gate
from bqskit.passes import GeneralizedSabreLayoutPass
from bqskit.passes import GeneralizedSabreRoutingPass
from bqskit.passes import GreedyPlacementPass
from bqskit.passes import SetModelPass
from bqskit.passes.mapping.apply import ApplyPlacement
from bqskit.qis import PermutationMatrix
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix

def test_simple(compiler: Compiler) -> None:
    cg = (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)
    model = MachineModel(8, cg)
    circuit = Circuit(5)
    for i in range(4):
        circuit.append_gate(CNOTGate(), (4, i))
        circuit.append_gate(
            U3Gate(), 4, U3Gate().calc_params(
                UnitaryMatrix.random(1),
            ),
        )
        circuit.append_gate(
            U3Gate(), i, U3Gate().calc_params(
                UnitaryMatrix.random(1),
            ),
        )
    circuit.append_gate(CNOTGate(), (0, 1))

    in_utry = circuit.get_unitary()

    workflow = [
        SetModelPass(model),
        GreedyPlacementPass(),
        GeneralizedSabreLayoutPass(),
        GeneralizedSabreRoutingPass(),
        ApplyPlacement(),
    ]

    cc, data = compiler.compile(circuit, workflow, True)
    pi = data['initial_mapping']
    pf = data['final_mapping']
    PI = PermutationMatrix.from_qubit_location(5, pi)
    PF = PermutationMatrix.from_qubit_location(5, pf)
    inactive = [i for i in range(cc.num_qudits) if i not in cc.active_qudits]
    inactive.sort(reverse=True)
    for i in inactive:
        cc.pop_qudit(i)
    assert cc.get_unitary().get_distance_from(PF.T @ in_utry @ PI) < 1e-7
    assert all(e in cg for e in cc.coupling_graph)
