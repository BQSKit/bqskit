from __future__ import annotations

from bqskit.compiler.compile import compile
from bqskit.compiler.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.qis.permutation import PermutationMatrix


def test_medium_circuit_compile(
    compiler: Compiler,
    optimization_level: int,
    medium_qasm_file: str,
) -> None:
    circuit = Circuit.from_file(medium_qasm_file)
    out_circuit, pi, pf = compile(
        circuit,
        optimization_level=optimization_level,
        with_mapping=True,
        compiler=compiler,
    )
    in_utry = circuit.get_unitary()
    out_utry = out_circuit.get_unitary()
    PI = PermutationMatrix.from_qubit_location(out_circuit.num_qudits, pi)
    PF = PermutationMatrix.from_qubit_location(out_circuit.num_qudits, pf)
    error = out_utry.get_distance_from(PF.T @ in_utry @ PI, 1)
    assert error <= 1e-8
