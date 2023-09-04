from __future__ import annotations

from bqskit.compiler.compile import compile
from bqskit.compiler.compiler import Compiler
from bqskit.compiler.machine import MachineModel
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import U3Gate
from bqskit.qis.graph import CouplingGraph
from bqskit.qis.permutation import PermutationMatrix


def test_compile_with_mapping(
        compiler: Compiler,
        optimization_level: int,
) -> None:
    circuit = Circuit(3)
    circuit.append_gate(U3Gate(), [0], [0, 1, 2])
    circuit.append_gate(U3Gate(), [1], [0, 1, 2])
    circuit.append_gate(U3Gate(), [2], [0, 1, 2])
    circuit.append_gate(CNOTGate(), [0, 1])
    circuit.append_gate(U3Gate(), [0], [0, 1, 2])
    circuit.append_gate(U3Gate(), [1], [0, 1, 2])
    circuit.append_gate(U3Gate(), [2], [0, 1, 2])
    circuit.append_gate(CNOTGate(), [1, 2])
    circuit.append_gate(U3Gate(), [0], [0, 1, 2])
    circuit.append_gate(U3Gate(), [1], [0, 1, 2])
    circuit.append_gate(U3Gate(), [2], [0, 1, 2])
    circuit.append_gate(CNOTGate(), [0, 2])
    circuit.append_gate(U3Gate(), [0], [0, 1, 2])
    circuit.append_gate(U3Gate(), [1], [0, 1, 2])
    circuit.append_gate(U3Gate(), [2], [0, 1, 2])
    correct_utry = circuit.get_unitary()

    coupling_graph = CouplingGraph([(i, i + 1) for i in range(5)] + [(3, 5)])
    model = MachineModel(6, coupling_graph)

    circuit, initial_mapping, final_mapping = compile(
        circuit,
        model,
        compiler=compiler,
        with_mapping=True,
        optimization_level=optimization_level,
    )

    assert len(initial_mapping) == 3
    assert len(final_mapping) == 3

    # This simple circuit shouldn't have moving pieces in the mapping
    assert set(initial_mapping) == set(final_mapping)

    # cut out the mapped subcircuit
    mapped_subcircuit = circuit.copy()
    inactive_qudits = set(range(circuit.num_qudits)) - set(initial_mapping)
    for qudit_index in sorted(inactive_qudits, reverse=True):
        mapped_subcircuit.pop_qudit(qudit_index)

    # convert global mappings to local permutations
    global_qudits = list(sorted(final_mapping))
    local_qudit_initial_perm = [global_qudits.index(i) for i in initial_mapping]
    local_qudit_final_perm = [global_qudits.index(i) for i in final_mapping]

    # undo the mapping and compare to the original unitary
    out_utry = mapped_subcircuit.get_unitary()
    PI = PermutationMatrix.from_qubit_location(3, local_qudit_initial_perm)
    PF = PermutationMatrix.from_qubit_location(3, local_qudit_final_perm)
    assert out_utry.get_distance_from(PF.T @ correct_utry @ PI) < 1e-6
