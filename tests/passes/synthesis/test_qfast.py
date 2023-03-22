from __future__ import annotations

from bqskit.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CircuitGate
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.passes import QFASTDecompositionPass
from bqskit.qis import UnitaryMatrix


class TestQFAST:
    def test_small_qubit(self) -> None:
        utry = UnitaryMatrix.random(2)
        circuit = Circuit.from_unitary(utry)
        qfast = QFASTDecompositionPass()
        circuit.perform(qfast)
        dist = circuit.get_unitary().get_distance_from(utry)
        assert dist <= 1e-5

    def test_small_qubit_with_compiler(self) -> None:
        with Compiler() as compiler:
            utry = UnitaryMatrix.random(2)
            circuit = Circuit.from_unitary(utry)
            qfast = QFASTDecompositionPass()
            circuit = compiler.compile(circuit, [qfast])
            dist = circuit.get_unitary().get_distance_from(utry)
            assert dist <= 1e-5

    def test_3_qubit(self) -> None:
        utry = UnitaryMatrix.random(3)
        circuit = Circuit.from_unitary(utry)
        qfast = QFASTDecompositionPass()
        circuit.perform(qfast)
        dist = circuit.get_unitary().get_distance_from(utry)
        assert dist <= 1e-5

    def test_3_qubit_with_compiler(self) -> None:
        with Compiler() as compiler:
            utry = UnitaryMatrix.random(3)
            circuit = Circuit.from_unitary(utry)
            qfast = QFASTDecompositionPass()
            circuit = compiler.compile(circuit, [qfast])
            dist = circuit.get_unitary().get_distance_from(utry)
            assert dist <= 1e-5

    def test_3_qubit_with_cnot_block(self) -> None:
        circuit = Circuit(2)
        circuit.append_gate(CNOTGate(), (0, 1))
        circuit.append_gate(U3Gate(), (0,))
        circuit.append_gate(U3Gate(), (1,))
        cg = CircuitGate(circuit)

        utry = UnitaryMatrix.random(3)
        circuit = Circuit.from_unitary(utry)
        qfast = QFASTDecompositionPass(cg)
        circuit.perform(qfast)
        dist = circuit.get_unitary().get_distance_from(utry)
        assert dist <= 1e-5
