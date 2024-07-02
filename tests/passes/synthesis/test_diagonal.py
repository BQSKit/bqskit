from __future__ import annotations

from numpy.random import normal

from scipy.linalg import expm

from bqskit.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.qis import UnitaryMatrix

from bqskit.qis.pauliz import PauliZMatrices
from bqskit.passes.synthesis.diagonal import DiagonalSynthesisPass


class TestDiagonalSynthesis:

    def test_1_qubit(self, compiler: Compiler) -> None:
        num_qubits = 1
        pauliz = PauliZMatrices(num_qubits)
        vector = [normal() for _ in range(len(pauliz))]
        H_matrix = pauliz.dot_product(vector)
        utry = UnitaryMatrix(expm(1j * H_matrix))

        circuit = Circuit.from_unitary(utry)
        synthesis = DiagonalSynthesisPass()
        circuit = compiler.compile(circuit, [synthesis])
        dist = circuit.get_unitary().get_distance_from(utry)

        assert dist <= 1e-5

    def test_2_qubit(self, compiler: Compiler) -> None:
        num_qubits = 2
        pauliz = PauliZMatrices(num_qubits)
        vector = [normal() for _ in range(len(pauliz))]
        H_matrix = pauliz.dot_product(vector)
        utry = UnitaryMatrix(expm(1j * H_matrix))

        circuit = Circuit.from_unitary(utry)
        synthesis = DiagonalSynthesisPass()
        circuit = compiler.compile(circuit, [synthesis])
        dist = circuit.get_unitary().get_distance_from(utry)

        assert dist <= 1e-5

    def test_3_qubit(self, compiler: Compiler) -> None:
        num_qubits = 3
        pauliz = PauliZMatrices(num_qubits)
        vector = [normal() for _ in range(len(pauliz))]
        H_matrix = pauliz.dot_product(vector)
        utry = UnitaryMatrix(expm(1j * H_matrix))

        circuit = Circuit.from_unitary(utry)
        synthesis = DiagonalSynthesisPass()
        circuit = compiler.compile(circuit, [synthesis])
        dist = circuit.get_unitary().get_distance_from(utry)

        assert dist <= 1e-5

    def test_4_qubit(self, compiler: Compiler) -> None:
        num_qubits = 4
        pauliz = PauliZMatrices(num_qubits)
        vector = [normal() for _ in range(len(pauliz))]
        H_matrix = pauliz.dot_product(vector)
        utry = UnitaryMatrix(expm(1j * H_matrix))

        circuit = Circuit.from_unitary(utry)
        synthesis = DiagonalSynthesisPass()
        circuit = compiler.compile(circuit, [synthesis])
        dist = circuit.get_unitary().get_distance_from(utry)

        assert dist <= 1e-5

    def test_5_qubit(self, compiler: Compiler) -> None:
        num_qubits = 5
        pauliz = PauliZMatrices(num_qubits)
        vector = [normal() for _ in range(len(pauliz))]
        H_matrix = pauliz.dot_product(vector)
        utry = UnitaryMatrix(expm(1j * H_matrix))

        circuit = Circuit.from_unitary(utry)
        synthesis = DiagonalSynthesisPass()
        circuit = compiler.compile(circuit, [synthesis])
        dist = circuit.get_unitary().get_distance_from(utry)

        assert dist <= 1e-5