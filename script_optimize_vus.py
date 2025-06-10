from bqskit.ir.circuit import Circuit, CircuitGate
from bqskit.ir.operation import Operation
from bqskit.ir.gates import VariableUnitaryGate
from bqskit.passes.synthesis.bzxz import BlockZXZPass
from bqskit.passes.processing.extract_diagonal import ExtractDiagonalPass
from bqskit.passes.synthesis.qsd import QSDPass
from bqskit.compiler.passdata import PassData

import asyncio

async def synthesize_unoptimized_gates(circuit: Circuit, min_qudit_size: int, perform_extract: bool, decompose_all: bool) -> Circuit:
    # Idea: get all VUs that have qubit counts greater than the min_qudit_size
    print(f"\n --------- SYNTHESIZING UNOPTIMIZED GATES... ---------\n")
    unitaries, points, locations = QSDPass.get_variable_unitary_pts(circuit, min_qudit_size)

    # initialize BZXZ pass and diagonal pass on these VUs
    bzxz = BlockZXZPass(min_qudit_size=min_qudit_size, decompose_all=decompose_all)
    diag = ExtractDiagonalPass(qudit_size=min_qudit_size)

    # run one round of BZXZ decomposition on set of unoptimized unitaries
    subcircs = []
    for u in unitaries:
        subcirc = bzxz.zxz(u)
        if perform_extract:
            await diag.run(subcirc, PassData(circuit=subcirc))
        subcircs.append(subcirc)


    # replace old VUs with new decompositions
    circ_gates = [CircuitGate(sc) for sc in subcircs]
    circ_ops = [
        Operation(g, loc, g._circuit.params)
        for g, loc in zip(circ_gates, locations)
    ]
    circuit.batch_replace(points, circ_ops)
    circuit.unfold_all()


    print("\n✅ Replaced all unoptimized VUs with decomposed circuits.")
    return circuit