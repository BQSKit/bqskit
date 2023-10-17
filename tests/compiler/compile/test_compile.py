from __future__ import annotations

from typing import Callable

import pytest

from bqskit.compiler.compile import compile
from bqskit.compiler.compiler import Compiler
from bqskit.compiler.machine import MachineModel
from bqskit.ir.circuit import Circuit
from bqskit.qis.graph import CouplingGraph
from bqskit.qis.permutation import PermutationMatrix


def default_model_gen(circuit: Circuit) -> MachineModel:
    """Generate a default model for the given circuit."""
    return MachineModel(circuit.num_qudits)


def linear_model_gen(circuit: Circuit) -> MachineModel:
    """Generate a linear model for the given circuit."""
    return MachineModel(
        circuit.num_qudits,
        CouplingGraph.linear(circuit.num_qudits),
    )


@pytest.mark.parametrize(
    'gen_model',
    [
        default_model_gen,
        linear_model_gen,
    ],
    ids=[
        'default',
        'linear',
    ],
)
def test_medium_circuit_compile(
    compiler: Compiler,
    optimization_level: int,
    medium_qasm_file: str,
    gen_model: Callable[[Circuit], MachineModel],
) -> None:
    circuit = Circuit.from_file(medium_qasm_file)
    model = gen_model(circuit)
    out_circuit, pi, pf = compile(
        circuit,
        optimization_level=optimization_level,
        with_mapping=True,
        compiler=compiler,
        model=model,
    )
    in_utry = circuit.get_unitary()
    out_utry = out_circuit.get_unitary()
    PI = PermutationMatrix.from_qubit_location(out_circuit.num_qudits, pi)
    PF = PermutationMatrix.from_qubit_location(out_circuit.num_qudits, pf)
    error = out_utry.get_distance_from(PF.T @ in_utry @ PI, 1)
    assert error <= 1e-8
    assert model.is_compatible(out_circuit)
