from __future__ import annotations

import logging

from bqskit.compiler.machine import MachineModel
from bqskit.compiler.passes import synthesis
from bqskit.compiler.passes.partitioning.scan import ScanPartitioner
from bqskit.compiler.passes.synthesis.leap import LEAPSynthesisPass
from bqskit.compiler.passes.util.unfold import UnfoldPass
from bqskit.compiler.passes.util.variabletou3 import VariableToU3Pass
from bqskit.compiler.search.generators.simple import SimpleLayerGenerator
from bqskit.ir import Circuit
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.ir.lang.qasm2.qasm2 import OPENQASM2Language

logging.getLogger('bqskit.compiler').setLevel(logging.DEBUG)


def mesh(
        n: int,
        m: int = None,
) -> set[tuple[int]]:
    """
    Generate a 2D mesh coupling map.

    Args:
            n (int): If only n is provided, then this is the side length of a square
                    grid. Otherwise it is the number of rows in the mesh.

            m (int|None): If m is provided, it is the number of columns in the mesh.

    Returns:
            coupling_map (set[tuple[int]]): The coupling map corresponding to the
                    2D nearest neighbor mesh that is nxn or nxm in dimensions.
    """
    cols = n if m is None else m
    rows = n

    edges = set()
    # Horizontals
    for i in range(rows):
        for j in range(cols - 1):
            edges.add((i * cols + j, i * cols + j + 1))
    # Verticals
    for i in range(rows - 1):
        for j in range(cols):
            edges.add((i * cols + j, i * cols + j + cols))
    return edges


def alltoall(num_q: int) -> set[tuple[int]]:
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


def linear(num_q: int) -> set[tuple[int]]:
    """
    Generate a linear coupling map.

    Args:
            num_q (int): Number of vertices in the graph.

    Returns:
            coupling_map (set[tuple[int]]): Linear couplings
    """
    return mesh(num_q, 1)


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
    mach = MachineModel(circ.get_size(), alltoall(num_q))
    data = {'machine_model': mach}
    part = ScanPartitioner(3)
    print('Partitioning circuit...')
    part.run(circ, data)

    # Layout

    # Synthesis
    print('Synthesizing circuit...')
    instantiate_options = {
        'min_iters': 0,
        'diff_tol_r': 1e-5,
        'dist_tol': 1e-11,
        'max_iters': 2500,
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
    unfolder = UnfoldPass()
    unfolder.run(circ, data)
    converter = VariableToU3Pass()
    converter.run(circ, data)

    print('Saving circuit...')
    with open('scratch/synthesized_' + filename, 'w') as f:
        f.write(OPENQASM2Language().encode(circ))
