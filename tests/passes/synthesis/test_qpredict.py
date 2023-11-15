from __future__ import annotations

from bqskit.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.passes import QPredictDecompositionPass
from bqskit.qis import UnitaryMatrix


class TestQPredict:

    def test_small_qubit(self, compiler: Compiler) -> None:
        utry = UnitaryMatrix.random(2)
        circuit = Circuit.from_unitary(utry)
        qpredict = QPredictDecompositionPass()
        circuit = compiler.compile(circuit, [qpredict])
        dist = circuit.get_unitary().get_distance_from(utry)
        assert dist <= 1e-3

    def test_3_qubit(self, compiler: Compiler) -> None:
        utry = UnitaryMatrix.random(3)
        circuit = Circuit.from_unitary(utry)
        qpredict = QPredictDecompositionPass()
        circuit = compiler.compile(circuit, [qpredict])
        dist = circuit.get_unitary().get_distance_from(utry)
        assert dist <= 1e-3
