from __future__ import annotations

import pytest

from bqskit import Circuit
from bqskit import MachineModel
from bqskit.ir.gates import CZGate
from bqskit.passes import GeneralizedSabreLayoutPass
from bqskit.passes import GeneralizedSabreRoutingPass
from bqskit.qis import CouplingGraph


def looping_circuit(
    uphill_swaps: int = 1,
    additional_local_minimum_gates: int = 0,
) -> Circuit:
    """
    Create a circuit that is arbitrarily hard for the advanced SABRE heuristics
    to route.

    The circuit is in a stable local minimum of the "lookahead" and "decay"
    SABRE heuristics, where you need to increase the values of those heuristics
    before you can make progress.  This minimum is stable no matter what
    weighting is attached to the cost of the gates in the extended set (except
    if they're weighted as zero).  If the decay reset interval is set
    sufficiently high, it _can_ become possible to escape the minimum, but you
    need to get arbitrarily lucky as the two inputs to this function are
    increased.

    Draw the circuit output from this function to see the hill structure.

    Args:
        uphill_swaps (int): how many swaps SABRE will need to make that increase
            the total score of the layout before it can make further progress
        additional_local_minimum_gates (int): how many additional gates to add
            in the front layer that produces the local minumum that the routing
            pass gets stuck in.  This increases the number of layouts that all
            have the same score that the pass will get stuck switching between.

    References:
        https://github.com/Qiskit/qiskit-terra/issues/7707
    """
    outers = 4 + additional_local_minimum_gates
    n_qubits = 2 * outers + 4 + uphill_swaps
    # This is (most of) the front layer, which is a bunch of outer qubits in the
    # coupling map.
    outer_pairs = [(i, n_qubits - i - 1) for i in range(outers)]
    inner_heuristic_peak = [
        # This gate is completely "inside" all the others in the front layer in
        # terms of the coupling map, so it's the only one that we can in theory
        # make progress towards without making the others worse.
        (outers + 1, outers + 2 + uphill_swaps),
        # These are the only two gates in the extended set, and they both get
        # further apart if you make a swap to bring the above gate closer
        # together, which is the trick that creates the "heuristic hill".
        (outers, outers + 1),
        (outers + 2 + uphill_swaps, outers + 3 + uphill_swaps),
    ]
    circuit = Circuit(n_qubits)
    for pair in outer_pairs + inner_heuristic_peak:
        circuit.append_gate(CZGate(), pair)
    return circuit


@pytest.mark.parametrize(
    ['swaps', 'local_mins'],
    [(1, 0), (2, 0), (2, 1), (3, 2)],
)
def test_escape_min_route(swaps: int, local_mins: int) -> None:
    circuit = looping_circuit(swaps, local_mins)
    cg = CouplingGraph.linear(circuit.num_qudits)
    data = {'machine_model': MachineModel(circuit.num_qudits, cg)}
    circuit.perform(GeneralizedSabreRoutingPass(), data)
    assert all(e in cg for e in circuit.coupling_graph)


@pytest.mark.parametrize(
    ['swaps', 'local_mins'],
    [(1, 0), (2, 0), (2, 1), (3, 2)],
)
def test_escape_min_layout(swaps: int, local_mins: int) -> None:
    circuit = looping_circuit(swaps, local_mins)
    cg = CouplingGraph.linear(circuit.num_qudits)
    data = {'machine_model': MachineModel(circuit.num_qudits, cg)}
    circuit.perform(GeneralizedSabreLayoutPass(3), data)
