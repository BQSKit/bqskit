from __future__ import annotations

import numpy as np

from bqskit.compiler.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.parameterized.rz import RZGate
from bqskit.passes import ZXZXZDecomposition
from bqskit.qis import UnitaryMatrix


def test_zxzxz_decomposition(compiler: Compiler) -> None:
    for i in range(100):
        test_utry = UnitaryMatrix.random(1)
        test_circuit = Circuit.from_unitary(test_utry)
        out_circuit = compiler.compile(test_circuit, ZXZXZDecomposition())
        assert out_circuit.get_unitary().get_distance_from(test_utry) < 5e-8


def test_zxzxz_decomposition_diagonal_collapses_gauge(
    compiler: Compiler,
) -> None:
    """A diagonal target's middle rotation is Clifford, so the two outer
    RZ gates are only constrained as a sum/difference -- how that total is
    split between them is a free gauge choice. Splitting it evenly, as the
    unconstrained formulas would, turns one gate needing synthesis into two.
    The decomposition should instead collapse the gauge onto a single gate,
    leaving the other an exact identity (angle 0)."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        theta = rng.uniform(-np.pi, np.pi)
        test_circuit = Circuit(1)
        test_circuit.append_gate(RZGate(), 0, [theta])
        test_utry = test_circuit.get_unitary()

        out_circuit = compiler.compile(test_circuit, ZXZXZDecomposition())
        assert out_circuit.get_unitary().get_distance_from(test_utry) < 5e-8

        angles = [
            op.params[0]
            for op in out_circuit
            if op.gate.num_params == 1
        ]
        assert any(abs(angle) < 1e-9 for angle in angles)
