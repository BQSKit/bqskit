from __future__ import annotations

from bqskit.compiler import CompilationTask
from bqskit.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.passes import QPredictDecompositionPass
from bqskit.qis import UnitaryMatrix


class TestQPredict:
    def test_small_qubit(self) -> None:
        utry = UnitaryMatrix.random(2)
        circuit = Circuit.from_unitary(utry)
        qpredict = QPredictDecompositionPass()
        circuit.perform(qpredict)
        dist = circuit.get_unitary().get_distance_from(utry)
        assert dist <= 1e-3

    def test_small_qubit_with_compiler(self, compiler: Compiler) -> None:
        utry = UnitaryMatrix.random(2)
        circuit = Circuit.from_unitary(utry)
        qpredict = QPredictDecompositionPass()
        circuit = compiler.compile(CompilationTask(circuit, [qpredict]))
        dist = circuit.get_unitary().get_distance_from(utry)
        assert dist <= 1e-3

    def test_3_qubit(self) -> None:
        utry = UnitaryMatrix.random(3)
        circuit = Circuit.from_unitary(utry)
        qpredict = QPredictDecompositionPass()
        circuit.perform(qpredict)
        dist = circuit.get_unitary().get_distance_from(utry)
        assert dist <= 1e-3

    def test_3_qubit_with_compiler(self, compiler: Compiler) -> None:
        utry = UnitaryMatrix.random(3)
        circuit = Circuit.from_unitary(utry)
        qpredict = QPredictDecompositionPass()
        circuit = compiler.compile(CompilationTask(circuit, [qpredict]))
        dist = circuit.get_unitary().get_distance_from(utry)
        assert dist <= 1e-3
