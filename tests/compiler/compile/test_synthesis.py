from __future__ import annotations

import pytest
pytest.importorskip('bqskit.ext')

from bqskit import compile
from bqskit import MachineModel
from bqskit.compiler.compiler import Compiler
from bqskit.compiler.machine import default_gate_set
from bqskit.ext.cirq.models import google_gate_set
from bqskit.ext.honeywell import honeywell_gate_set
from bqskit.ext.rigetti import rigetti_gate_set
from bqskit.ir.gate import Gate
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import IToffoliGate
from bqskit.ir.gates import PhasedXZGate
from bqskit.ir.gates import ToffoliGate
from bqskit.ir.gates import U1qGate
from bqskit.ir.gates import U3Gate
from bqskit.ir.gates import XGate
from bqskit.qis import UnitaryMatrix


@pytest.mark.parametrize('sq_utry', [UnitaryMatrix.random(1) for i in range(5)])
@pytest.mark.parametrize(
    'gate_set', [
        {U3Gate()},
        {PhasedXZGate()},
        {U1qGate(), XGate()},
        default_gate_set,
        rigetti_gate_set,
        honeywell_gate_set,
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
    assert out_circuit.get_unitary().get_distance_from(sq_utry, 1) < 1e-10


@pytest.mark.parametrize('tq_utry', [UnitaryMatrix.random(2) for i in range(5)])
@pytest.mark.parametrize(
    'gate_set', [
        default_gate_set,
        rigetti_gate_set,
        honeywell_gate_set,
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
    assert out_circuit.get_unitary().get_distance_from(tq_utry, 1) < 1e-10


@pytest.mark.parametrize(
    'gate_set', [
        default_gate_set,
        rigetti_gate_set,
        honeywell_gate_set,
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
    utry = toffoli_unitary
    assert out_circuit.get_unitary().get_distance_from(utry, 1) < 1e-10


def test_fail_on_larger_max_synthesis_size(compiler: Compiler) -> None:
    utry = UnitaryMatrix.random(4)
    with pytest.raises(ValueError):
        compile(utry, max_synthesis_size=3, compiler=compiler)
