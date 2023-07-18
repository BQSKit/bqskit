from __future__ import annotations

from bqskit.compiler.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.passes import ZXZXZDecomposition
from bqskit.qis import UnitaryMatrix


def test_zxzxz_decomposition(compiler: Compiler) -> None:
    for i in range(100):
        test_utry = UnitaryMatrix.random(1)
        test_circuit = Circuit.from_unitary(test_utry)
        out_circuit = compiler.compile(test_circuit, ZXZXZDecomposition())
        assert out_circuit.get_unitary().get_distance_from(test_utry) < 5e-8
