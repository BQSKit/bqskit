from __future__ import annotations

import itertools as it

import pytest

from bqskit import compile
from bqskit import MachineModel
from bqskit.compiler.compiler import Compiler
from bqskit.compiler.gateset import GateSet
from bqskit.ext.cirq.models import google_gate_set
from bqskit.ext.quantinuum import quantinuum_gate_set
from bqskit.ext.rigetti import rigetti_gate_set
from bqskit.ir.gate import Gate
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import IToffoliGate
from bqskit.ir.gates import PhasedXZGate
from bqskit.ir.gates import ToffoliGate
from bqskit.ir.gates import U1qGate
from bqskit.ir.gates import U3Gate
from bqskit.ir.gates import XGate
from bqskit.qis import PermutationMatrix
from bqskit.qis import UnitaryMatrix


def get_distance_from_pa(U: UnitaryMatrix, V: UnitaryMatrix) -> float:
    """Get distance between U and V up to a permutation."""
    width = U.num_qudits
    perms = list(it.permutations(range(width)))
    Pis = [PermutationMatrix.from_qubit_location(width, p) for p in perms]
    Pos = [PermutationMatrix.from_qubit_location(width, p) for p in perms]
    dists = [
        U.get_distance_from(Po.T @ V @ Pi, 1)
        for Pi, Po in it.product(Pis, Pos)
    ]
    return min(dists)


@pytest.mark.parametrize('sq_utry', [UnitaryMatrix.random(1) for i in range(5)])
@pytest.mark.parametrize(
    'gate_set', [
        {U3Gate()},
        {PhasedXZGate()},
        {U1qGate(), XGate()},
        GateSet.default_gate_set(),
        rigetti_gate_set,
        quantinuum_gate_set,
        google_gate_set,
        {IToffoliGate(), U3Gate()},
        {IToffoliGate(), CNOTGate(), U3Gate()},
    ],
)
def test_single_qudit_synthesis(
    sq_utry: UnitaryMatrix,
    optimization_level: int,
    gate_set: set[Gate],
    compiler: Compiler,
) -> None:
    out_circuit = compile(
        sq_utry,
        model=MachineModel(1, gate_set=gate_set),
        optimization_level=optimization_level,
        compiler=compiler,
    )
    if U3Gate() in gate_set:
        assert out_circuit.num_operations == 1
    assert out_circuit.num_qudits == 1
    assert len(out_circuit.gate_set.difference(gate_set)) == 0
    assert out_circuit.get_unitary().get_distance_from(sq_utry, 1) < 1e-8


@pytest.mark.parametrize('tq_utry', [UnitaryMatrix.random(2) for i in range(5)])
@pytest.mark.parametrize(
    'gate_set', [
        GateSet.default_gate_set(),
        rigetti_gate_set,
        quantinuum_gate_set,
        google_gate_set,
        {IToffoliGate(), CNOTGate(), U3Gate()},
    ],
)
def test_two_qudit_synthesis(
    tq_utry: UnitaryMatrix,
    optimization_level: int,
    gate_set: set[Gate],
    compiler: Compiler,
) -> None:
    out_circuit = compile(
        tq_utry,
        model=MachineModel(2, gate_set=gate_set),
        optimization_level=optimization_level,
        compiler=compiler,
    )
    assert out_circuit.num_qudits == 2
    assert len(out_circuit.gate_set.difference(gate_set)) == 0
    out_utry = out_circuit.get_unitary()

    if optimization_level == 4:
        assert get_distance_from_pa(out_utry, tq_utry) < 1e-8

    else:
        assert out_utry.get_distance_from(tq_utry, 1) < 1e-8


@pytest.mark.parametrize(
    'gate_set', [
        GateSet.default_gate_set(),
        rigetti_gate_set,
        quantinuum_gate_set,
        {IToffoliGate(), U3Gate()},
        {ToffoliGate(), U3Gate()},
    ],
)
def test_three_qudit_synthesis(
    toffoli_unitary: UnitaryMatrix,
    optimization_level: int,
    gate_set: set[Gate],
    compiler: Compiler,
) -> None:
    out_circuit = compile(
        toffoli_unitary,
        model=MachineModel(3, gate_set=gate_set),
        optimization_level=optimization_level,
        compiler=compiler,
    )
    assert out_circuit.num_qudits == 3
    assert len(out_circuit.gate_set.difference(gate_set)) == 0
    out_utry = out_circuit.get_unitary()

    if optimization_level == 4:
        assert get_distance_from_pa(out_utry, toffoli_unitary) < 1e-8

    else:
        assert out_utry.get_distance_from(toffoli_unitary, 1) < 1e-8


def test_fail_on_larger_max_synthesis_size(compiler: Compiler) -> None:
    utry = UnitaryMatrix.random(4)
    with pytest.raises(ValueError):
        compile(utry, max_synthesis_size=3, compiler=compiler)


@pytest.mark.parametrize('dim', [2, 4, 8])
def test_identity_synthesis(
    optimization_level: int,
    dim: int,
    compiler: Compiler,
) -> None:
    out_circuit = compile(
        UnitaryMatrix.identity(dim),
        optimization_level=optimization_level,
        compiler=compiler,
    )
    assert out_circuit.get_unitary().get_distance_from(
        UnitaryMatrix.identity(dim), 1,
    ) < 1e-8

    # TODO: Re-enable this check when tree gate deletion hits the OTS.
    # In cases where the identity is synthesized to two cnots surrounded
    # by a bunch of single-qudit gates, scanning gate removal cannot
    # remove either cnot.
    # if optimization_level >= 3:
    #     assert out_circuit.num_operations <= 3


@pytest.mark.parametrize('num_qudits', [1, 2])
def test_qutrit_synthesis(optimization_level: int, num_qudits: int) -> None:
    in_utry = UnitaryMatrix.random(num_qudits, [3] * num_qudits)
    out_circuit = compile(in_utry, optimization_level=optimization_level)
    assert out_circuit.get_unitary().get_distance_from(in_utry, 1) < 1e-8


@pytest.mark.parametrize('num_qudits', [1])
def test_ququart_synthesis(optimization_level: int, num_qudits: int) -> None:
    in_utry = UnitaryMatrix.random(num_qudits, [4] * num_qudits)
    out_circuit = compile(in_utry, optimization_level=optimization_level)
    assert out_circuit.get_unitary().get_distance_from(in_utry, 1) < 1e-8
