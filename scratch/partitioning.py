from __future__ import annotations

import logging

from bqskit.compiler.machine import MachineModel
from bqskit.compiler.passes.partitioning.scan import ScanPartitioner
from bqskit.compiler.passes.synthesis.leap import LEAPSynthesisPass
from bqskit.compiler.passes.util.unfold import UnfoldPass
from bqskit.compiler.passes.util.variabletou3 import VariableToU3Pass
from bqskit.compiler.search.generators.simple import SimpleLayerGenerator
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.ir.lang.qasm2.qasm2 import OPENQASM2Language
from bqskit.compiler.passes.util.intermediate import SaveIntermediatePass

logging.getLogger('bqskit').setLevel(logging.DEBUG)


def alltoall(num_q: int) -> set[tuple[int, int]]:
    """
    Generate an all to all coupling map.

    Args:
            num_q (int): Number of vertices in the graph.

    Returns:
            coupling_map (set[tuple[int]]): All to all couplings.
    """
    edges = set()
    for i in range(num_q):
        for j in range(num_q):
            if i != j:
                edges.add((i, j))
    return edges


if __name__ == '__main__':

    filename = 'qft_5.qasm'
    with open('scratch/qft_qasm/' + filename) as f:
        circ = OPENQASM2Language().decode(f.read())

    with open('scratch/synthesized_' + filename) as f:
        syncirc = OPENQASM2Language().decode(f.read())
    print(circ.get_unitary().get_distance_from(syncirc.get_unitary()))

    # Prepare circuit
    print('Loading circuit...')
    with open('scratch/qft_qasm/' + filename) as f:
        circ = OPENQASM2Language().decode(f.read())
    num_q = circ.get_size()
    # Run partitioner on logical topology
    mach = MachineModel(circ.get_size(), list(alltoall(num_q)))
    data = {'machine_model': mach}
    part = ScanPartitioner(3)
    print('Partitioning circuit...')
    part.run(circ, data)

    pre_saver = SaveIntermediatePass("./scratch/temp/", "pre_qft_5")
    pre_saver.run(circ, {})

    # Synthesis
    print('Synthesizing circuit...')
    instantiate_options = {
        'min_iters': 0,
        'diff_tol_r': 1e-3,
        'dist_tol': 1e-10,
        'max_iters': 1000,
    }
    layer_generator = SimpleLayerGenerator(
        single_qudit_gate_1=VariableUnitaryGate(1),
    )
    synthesizer = LEAPSynthesisPass(
        layer_generator=layer_generator,
        instantiate_options=instantiate_options,
    )
    synthesizer.run(circ, data)

    print('Transforming circuit...')
    post_saver = SaveIntermediatePass("./scratch/temp/", "post_qft_5")
    post_saver.run(circ, {})

    unfolder = UnfoldPass()
    unfolder.run(circ, data)
    converter = VariableToU3Pass()
    converter.run(circ, data)


    print('Saving circuit...')
    with open('scratch/synthesized_' + filename, 'w') as f:
        f.write(OPENQASM2Language().encode(circ))
