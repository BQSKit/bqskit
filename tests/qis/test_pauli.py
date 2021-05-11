"""This module tests the pauli library in bqskit.qis.pauli."""
from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np
import pytest

from bqskit.qis.pauli import PauliMatrices


class TestPauliMatricesConstructor:

    def in_array(self, needle: Any, haystack: Any) -> bool:
        for elem in haystack:
            if np.allclose(elem, needle):
                return True

        return False

    def test_invalid_type(self, not_an_int: Any) -> None:
        with pytest.raises(TypeError):
            paulis = PauliMatrices(not_an_int)  # noqa

    @pytest.mark.parametrize('size', [-10, -5, 0])
    def test_invalid_value(self, size: int) -> None:
        with pytest.raises(ValueError):
            paulis = PauliMatrices(size)  # noqa

    def test_size_1(self) -> None:
        num_qubits = 1
        paulis = PauliMatrices(num_qubits)
        assert len(paulis) == 4 ** num_qubits

        I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        assert self.in_array(I, paulis)
        assert self.in_array(X, paulis)
        assert self.in_array(Y, paulis)
        assert self.in_array(Z, paulis)

    def test_size_2(self) -> None:
        num_qubits = 2
        paulis = PauliMatrices(num_qubits)
        assert len(paulis) == 4 ** num_qubits

        I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        assert self.in_array(np.kron(X, X), paulis)
        assert self.in_array(np.kron(X, Y), paulis)
        assert self.in_array(np.kron(X, Z), paulis)
        assert self.in_array(np.kron(X, I), paulis)
        assert self.in_array(np.kron(Y, X), paulis)
        assert self.in_array(np.kron(Y, Y), paulis)
        assert self.in_array(np.kron(Y, Z), paulis)
        assert self.in_array(np.kron(Y, I), paulis)
        assert self.in_array(np.kron(Z, X), paulis)
        assert self.in_array(np.kron(Z, Y), paulis)
        assert self.in_array(np.kron(Z, Z), paulis)
        assert self.in_array(np.kron(Z, I), paulis)
        assert self.in_array(np.kron(I, X), paulis)
        assert self.in_array(np.kron(I, Y), paulis)
        assert self.in_array(np.kron(I, Z), paulis)
        assert self.in_array(np.kron(I, I), paulis)

    def test_size_3(self) -> None:
        num_qubits = 3
        paulis = PauliMatrices(num_qubits)
        assert len(paulis) == 4 ** num_qubits

        I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        assert self.in_array(np.kron(X, np.kron(X, X)), paulis)
        assert self.in_array(np.kron(X, np.kron(X, Y)), paulis)
        assert self.in_array(np.kron(X, np.kron(X, Z)), paulis)
        assert self.in_array(np.kron(X, np.kron(X, I)), paulis)
        assert self.in_array(np.kron(X, np.kron(Y, X)), paulis)
        assert self.in_array(np.kron(X, np.kron(Y, Y)), paulis)
        assert self.in_array(np.kron(X, np.kron(Y, Z)), paulis)
        assert self.in_array(np.kron(X, np.kron(Y, I)), paulis)
        assert self.in_array(np.kron(X, np.kron(Z, X)), paulis)
        assert self.in_array(np.kron(X, np.kron(Z, Y)), paulis)
        assert self.in_array(np.kron(X, np.kron(Z, Z)), paulis)
        assert self.in_array(np.kron(X, np.kron(Z, I)), paulis)
        assert self.in_array(np.kron(X, np.kron(I, X)), paulis)
        assert self.in_array(np.kron(X, np.kron(I, Y)), paulis)
        assert self.in_array(np.kron(X, np.kron(I, Z)), paulis)
        assert self.in_array(np.kron(X, np.kron(I, I)), paulis)
        assert self.in_array(np.kron(Y, np.kron(X, X)), paulis)
        assert self.in_array(np.kron(Y, np.kron(X, Y)), paulis)
        assert self.in_array(np.kron(Y, np.kron(X, Z)), paulis)
        assert self.in_array(np.kron(Y, np.kron(X, I)), paulis)
        assert self.in_array(np.kron(Y, np.kron(Y, X)), paulis)
        assert self.in_array(np.kron(Y, np.kron(Y, Y)), paulis)
        assert self.in_array(np.kron(Y, np.kron(Y, Z)), paulis)
        assert self.in_array(np.kron(Y, np.kron(Y, I)), paulis)
        assert self.in_array(np.kron(Y, np.kron(Z, X)), paulis)
        assert self.in_array(np.kron(Y, np.kron(Z, Y)), paulis)
        assert self.in_array(np.kron(Y, np.kron(Z, Z)), paulis)
        assert self.in_array(np.kron(Y, np.kron(Z, I)), paulis)
        assert self.in_array(np.kron(Y, np.kron(I, X)), paulis)
        assert self.in_array(np.kron(Y, np.kron(I, Y)), paulis)
        assert self.in_array(np.kron(Y, np.kron(I, Z)), paulis)
        assert self.in_array(np.kron(Y, np.kron(I, I)), paulis)
        assert self.in_array(np.kron(Z, np.kron(X, X)), paulis)
        assert self.in_array(np.kron(Z, np.kron(X, Y)), paulis)
        assert self.in_array(np.kron(Z, np.kron(X, Z)), paulis)
        assert self.in_array(np.kron(Z, np.kron(X, I)), paulis)
        assert self.in_array(np.kron(Z, np.kron(Y, X)), paulis)
        assert self.in_array(np.kron(Z, np.kron(Y, Y)), paulis)
        assert self.in_array(np.kron(Z, np.kron(Y, Z)), paulis)
        assert self.in_array(np.kron(Z, np.kron(Y, I)), paulis)
        assert self.in_array(np.kron(Z, np.kron(Z, X)), paulis)
        assert self.in_array(np.kron(Z, np.kron(Z, Y)), paulis)
        assert self.in_array(np.kron(Z, np.kron(Z, Z)), paulis)
        assert self.in_array(np.kron(Z, np.kron(Z, I)), paulis)
        assert self.in_array(np.kron(Z, np.kron(I, X)), paulis)
        assert self.in_array(np.kron(Z, np.kron(I, Y)), paulis)
        assert self.in_array(np.kron(Z, np.kron(I, Z)), paulis)
        assert self.in_array(np.kron(Z, np.kron(I, I)), paulis)
        assert self.in_array(np.kron(I, np.kron(X, X)), paulis)
        assert self.in_array(np.kron(I, np.kron(X, Y)), paulis)
        assert self.in_array(np.kron(I, np.kron(X, Z)), paulis)
        assert self.in_array(np.kron(I, np.kron(X, I)), paulis)
        assert self.in_array(np.kron(I, np.kron(Y, X)), paulis)
        assert self.in_array(np.kron(I, np.kron(Y, Y)), paulis)
        assert self.in_array(np.kron(I, np.kron(Y, Z)), paulis)
        assert self.in_array(np.kron(I, np.kron(Y, I)), paulis)
        assert self.in_array(np.kron(I, np.kron(Z, X)), paulis)
        assert self.in_array(np.kron(I, np.kron(Z, Y)), paulis)
        assert self.in_array(np.kron(I, np.kron(Z, Z)), paulis)
        assert self.in_array(np.kron(I, np.kron(Z, I)), paulis)
        assert self.in_array(np.kron(I, np.kron(I, X)), paulis)
        assert self.in_array(np.kron(I, np.kron(I, Y)), paulis)
        assert self.in_array(np.kron(I, np.kron(I, Z)), paulis)
        assert self.in_array(np.kron(I, np.kron(I, I)), paulis)


class TestPauliMatricesGetProjectionMatrices:

    def in_array(self, needle: Any, haystack: Any) -> bool:
        for elem in haystack:
            if np.allclose(elem, needle):
                return True

        return False

    @pytest.mark.parametrize('invalid_qubit', ['a', 1.37, False, [1.37]])
    def test_invalid_type_1(self, invalid_qubit: Any) -> None:
        paulis = PauliMatrices(4)
        with pytest.raises(TypeError):
            projected_paulis = paulis.get_projection_matrices(invalid_qubit)  # noqa

    def test_invalid_type_2(self, not_an_int: Any) -> None:
        paulis = PauliMatrices(4)
        with pytest.raises(TypeError):
            projected_paulis = paulis.get_projection_matrices([not_an_int])  # noqa

    @pytest.mark.parametrize('invalid_qubit', [-5, -2, 4, 10])
    def test_invalid_value_1(self, invalid_qubit: int) -> None:
        paulis = PauliMatrices(4)
        with pytest.raises(ValueError):
            projected_paulis = paulis.get_projection_matrices([invalid_qubit])  # noqa

    @pytest.mark.parametrize('invalid_q_set', [[], [0, 0], [0, 1, 2, 4]])
    def test_invalid_value_2(self, invalid_q_set: list[int]) -> None:
        paulis = PauliMatrices(4)
        with pytest.raises(ValueError):
            paulis.get_projection_matrices(invalid_q_set)

    def test_proj_3_0(self) -> None:
        num_qubits = 3
        qubit_proj = 0
        paulis = PauliMatrices(num_qubits)
        projs = paulis.get_projection_matrices([qubit_proj])
        assert len(projs) == 4

        I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        assert self.in_array(np.kron(np.kron(X, I), I), projs)
        assert self.in_array(np.kron(np.kron(Y, I), I), projs)
        assert self.in_array(np.kron(np.kron(Z, I), I), projs)
        assert self.in_array(np.kron(np.kron(I, I), I), projs)

    def test_proj_3_1(self) -> None:
        num_qubits = 3
        qubit_proj = 1
        paulis = PauliMatrices(num_qubits)
        projs = paulis.get_projection_matrices([qubit_proj])
        assert len(projs) == 4

        I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        assert self.in_array(np.kron(np.kron(I, X), I), projs)
        assert self.in_array(np.kron(np.kron(I, Y), I), projs)
        assert self.in_array(np.kron(np.kron(I, Z), I), projs)
        assert self.in_array(np.kron(np.kron(I, I), I), projs)

    def test_proj_3_2(self) -> None:
        num_qubits = 3
        qubit_proj = 2
        paulis = PauliMatrices(num_qubits)
        projs = paulis.get_projection_matrices([qubit_proj])
        assert len(projs) == 4

        I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        assert self.in_array(np.kron(np.kron(I, I), X), projs)
        assert self.in_array(np.kron(np.kron(I, I), Y), projs)
        assert self.in_array(np.kron(np.kron(I, I), Z), projs)
        assert self.in_array(np.kron(np.kron(I, I), I), projs)

    def test_proj_4_0(self) -> None:
        num_qubits = 4
        qubit_proj = 0
        paulis = PauliMatrices(num_qubits)
        projs = paulis.get_projection_matrices([qubit_proj])
        assert len(projs) == 4

        I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        assert self.in_array(np.kron(np.kron(np.kron(X, I), I), I), projs)
        assert self.in_array(np.kron(np.kron(np.kron(Y, I), I), I), projs)
        assert self.in_array(np.kron(np.kron(np.kron(Z, I), I), I), projs)
        assert self.in_array(np.kron(np.kron(np.kron(I, I), I), I), projs)

    def test_proj_4_1(self) -> None:
        num_qubits = 4
        qubit_proj = 1
        paulis = PauliMatrices(num_qubits)
        projs = paulis.get_projection_matrices([qubit_proj])
        assert len(projs) == 4

        I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        assert self.in_array(np.kron(np.kron(np.kron(I, X), I), I), projs)
        assert self.in_array(np.kron(np.kron(np.kron(I, Y), I), I), projs)
        assert self.in_array(np.kron(np.kron(np.kron(I, Z), I), I), projs)
        assert self.in_array(np.kron(np.kron(np.kron(I, I), I), I), projs)

    def test_proj_4_2(self) -> None:
        num_qubits = 4
        qubit_proj = 2
        paulis = PauliMatrices(num_qubits)
        projs = paulis.get_projection_matrices([qubit_proj])
        assert len(projs) == 4

        I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        assert self.in_array(np.kron(np.kron(np.kron(I, I), X), I), projs)
        assert self.in_array(np.kron(np.kron(np.kron(I, I), Y), I), projs)
        assert self.in_array(np.kron(np.kron(np.kron(I, I), Z), I), projs)
        assert self.in_array(np.kron(np.kron(np.kron(I, I), I), I), projs)

    def test_proj_4_3(self) -> None:
        num_qubits = 4
        qubit_proj = 3
        paulis = PauliMatrices(num_qubits)
        projs = paulis.get_projection_matrices([qubit_proj])
        assert len(projs) == 4

        I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        assert self.in_array(np.kron(np.kron(np.kron(I, I), I), X), projs)
        assert self.in_array(np.kron(np.kron(np.kron(I, I), I), Y), projs)
        assert self.in_array(np.kron(np.kron(np.kron(I, I), I), Z), projs)
        assert self.in_array(np.kron(np.kron(np.kron(I, I), I), I), projs)

    def test_proj_3_01(self) -> None:
        num_qubits = 3
        qubit_pro1 = 0
        qubit_pro2 = 1
        paulis = PauliMatrices(num_qubits)
        projs = paulis.get_projection_matrices([qubit_pro1, qubit_pro2])
        assert len(projs) == 16

        I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        assert self.in_array(np.kron(np.kron(X, I), I), projs)
        assert self.in_array(np.kron(np.kron(Y, I), I), projs)
        assert self.in_array(np.kron(np.kron(Z, I), I), projs)
        assert self.in_array(np.kron(np.kron(I, I), I), projs)
        assert self.in_array(np.kron(np.kron(X, X), I), projs)
        assert self.in_array(np.kron(np.kron(Y, X), I), projs)
        assert self.in_array(np.kron(np.kron(Z, X), I), projs)
        assert self.in_array(np.kron(np.kron(I, X), I), projs)
        assert self.in_array(np.kron(np.kron(X, Y), I), projs)
        assert self.in_array(np.kron(np.kron(Y, Y), I), projs)
        assert self.in_array(np.kron(np.kron(Z, Y), I), projs)
        assert self.in_array(np.kron(np.kron(I, Y), I), projs)
        assert self.in_array(np.kron(np.kron(X, Z), I), projs)
        assert self.in_array(np.kron(np.kron(Y, Z), I), projs)
        assert self.in_array(np.kron(np.kron(Z, Z), I), projs)
        assert self.in_array(np.kron(np.kron(I, Z), I), projs)

    def test_proj_3_02(self) -> None:
        num_qubits = 3
        qubit_pro1 = 0
        qubit_pro2 = 2
        paulis = PauliMatrices(num_qubits)
        projs = paulis.get_projection_matrices([qubit_pro1, qubit_pro2])
        assert len(projs) == 16

        I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        assert self.in_array(np.kron(np.kron(X, I), I), projs)
        assert self.in_array(np.kron(np.kron(Y, I), I), projs)
        assert self.in_array(np.kron(np.kron(Z, I), I), projs)
        assert self.in_array(np.kron(np.kron(I, I), I), projs)
        assert self.in_array(np.kron(np.kron(X, I), X), projs)
        assert self.in_array(np.kron(np.kron(Y, I), X), projs)
        assert self.in_array(np.kron(np.kron(Z, I), X), projs)
        assert self.in_array(np.kron(np.kron(I, I), X), projs)
        assert self.in_array(np.kron(np.kron(X, I), Y), projs)
        assert self.in_array(np.kron(np.kron(Y, I), Y), projs)
        assert self.in_array(np.kron(np.kron(Z, I), Y), projs)
        assert self.in_array(np.kron(np.kron(I, I), Y), projs)
        assert self.in_array(np.kron(np.kron(X, I), Z), projs)
        assert self.in_array(np.kron(np.kron(Y, I), Z), projs)
        assert self.in_array(np.kron(np.kron(Z, I), Z), projs)
        assert self.in_array(np.kron(np.kron(I, I), Z), projs)

    def test_proj_3_12(self) -> None:
        num_qubits = 3
        qubit_pro1 = 1
        qubit_pro2 = 2
        paulis = PauliMatrices(num_qubits)
        projs = paulis.get_projection_matrices([qubit_pro1, qubit_pro2])
        assert len(projs) == 16

        I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        assert self.in_array(np.kron(np.kron(I, X), I), projs)
        assert self.in_array(np.kron(np.kron(I, Y), I), projs)
        assert self.in_array(np.kron(np.kron(I, Z), I), projs)
        assert self.in_array(np.kron(np.kron(I, I), I), projs)
        assert self.in_array(np.kron(np.kron(I, X), X), projs)
        assert self.in_array(np.kron(np.kron(I, Y), X), projs)
        assert self.in_array(np.kron(np.kron(I, Z), X), projs)
        assert self.in_array(np.kron(np.kron(I, I), X), projs)
        assert self.in_array(np.kron(np.kron(I, X), Y), projs)
        assert self.in_array(np.kron(np.kron(I, Y), Y), projs)
        assert self.in_array(np.kron(np.kron(I, Z), Y), projs)
        assert self.in_array(np.kron(np.kron(I, I), Y), projs)
        assert self.in_array(np.kron(np.kron(I, X), Z), projs)
        assert self.in_array(np.kron(np.kron(I, Y), Z), projs)
        assert self.in_array(np.kron(np.kron(I, Z), Z), projs)
        assert self.in_array(np.kron(np.kron(I, I), Z), projs)

    def test_proj_3_012(self) -> None:
        num_qubits = 3
        paulis = PauliMatrices(num_qubits)
        projs = paulis.get_projection_matrices([0, 1, 2])
        assert len(projs) == 64

        I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        assert self.in_array(np.kron(np.kron(I, X), I), projs)
        assert self.in_array(np.kron(np.kron(I, Y), I), projs)
        assert self.in_array(np.kron(np.kron(I, Z), I), projs)
        assert self.in_array(np.kron(np.kron(I, I), I), projs)
        assert self.in_array(np.kron(np.kron(I, X), X), projs)
        assert self.in_array(np.kron(np.kron(I, Y), X), projs)
        assert self.in_array(np.kron(np.kron(I, Z), X), projs)
        assert self.in_array(np.kron(np.kron(I, I), X), projs)
        assert self.in_array(np.kron(np.kron(I, X), Y), projs)
        assert self.in_array(np.kron(np.kron(I, Y), Y), projs)
        assert self.in_array(np.kron(np.kron(I, Z), Y), projs)
        assert self.in_array(np.kron(np.kron(I, I), Y), projs)
        assert self.in_array(np.kron(np.kron(I, X), Z), projs)
        assert self.in_array(np.kron(np.kron(I, Y), Z), projs)
        assert self.in_array(np.kron(np.kron(I, Z), Z), projs)
        assert self.in_array(np.kron(np.kron(I, I), Z), projs)

        assert self.in_array(np.kron(np.kron(X, X), I), projs)
        assert self.in_array(np.kron(np.kron(X, Y), I), projs)
        assert self.in_array(np.kron(np.kron(X, Z), I), projs)
        assert self.in_array(np.kron(np.kron(X, I), I), projs)
        assert self.in_array(np.kron(np.kron(X, X), X), projs)
        assert self.in_array(np.kron(np.kron(X, Y), X), projs)
        assert self.in_array(np.kron(np.kron(X, Z), X), projs)
        assert self.in_array(np.kron(np.kron(X, I), X), projs)
        assert self.in_array(np.kron(np.kron(X, X), Y), projs)
        assert self.in_array(np.kron(np.kron(X, Y), Y), projs)
        assert self.in_array(np.kron(np.kron(X, Z), Y), projs)
        assert self.in_array(np.kron(np.kron(X, I), Y), projs)
        assert self.in_array(np.kron(np.kron(X, X), Z), projs)
        assert self.in_array(np.kron(np.kron(X, Y), Z), projs)
        assert self.in_array(np.kron(np.kron(X, Z), Z), projs)
        assert self.in_array(np.kron(np.kron(X, I), Z), projs)

        assert self.in_array(np.kron(np.kron(Y, X), I), projs)
        assert self.in_array(np.kron(np.kron(Y, Y), I), projs)
        assert self.in_array(np.kron(np.kron(Y, Z), I), projs)
        assert self.in_array(np.kron(np.kron(Y, I), I), projs)
        assert self.in_array(np.kron(np.kron(Y, X), X), projs)
        assert self.in_array(np.kron(np.kron(Y, Y), X), projs)
        assert self.in_array(np.kron(np.kron(Y, Z), X), projs)
        assert self.in_array(np.kron(np.kron(Y, I), X), projs)
        assert self.in_array(np.kron(np.kron(Y, X), Y), projs)
        assert self.in_array(np.kron(np.kron(Y, Y), Y), projs)
        assert self.in_array(np.kron(np.kron(Y, Z), Y), projs)
        assert self.in_array(np.kron(np.kron(Y, I), Y), projs)
        assert self.in_array(np.kron(np.kron(Y, X), Z), projs)
        assert self.in_array(np.kron(np.kron(Y, Y), Z), projs)
        assert self.in_array(np.kron(np.kron(Y, Z), Z), projs)
        assert self.in_array(np.kron(np.kron(Y, I), Z), projs)

        assert self.in_array(np.kron(np.kron(Z, X), I), projs)
        assert self.in_array(np.kron(np.kron(Z, Y), I), projs)
        assert self.in_array(np.kron(np.kron(Z, Z), I), projs)
        assert self.in_array(np.kron(np.kron(Z, I), I), projs)
        assert self.in_array(np.kron(np.kron(Z, X), X), projs)
        assert self.in_array(np.kron(np.kron(Z, Y), X), projs)
        assert self.in_array(np.kron(np.kron(Z, Z), X), projs)
        assert self.in_array(np.kron(np.kron(Z, I), X), projs)
        assert self.in_array(np.kron(np.kron(Z, X), Y), projs)
        assert self.in_array(np.kron(np.kron(Z, Y), Y), projs)
        assert self.in_array(np.kron(np.kron(Z, Z), Y), projs)
        assert self.in_array(np.kron(np.kron(Z, I), Y), projs)
        assert self.in_array(np.kron(np.kron(Z, X), Z), projs)
        assert self.in_array(np.kron(np.kron(Z, Y), Z), projs)
        assert self.in_array(np.kron(np.kron(Z, Z), Z), projs)
        assert self.in_array(np.kron(np.kron(Z, I), Z), projs)

    def test_proj_4_02(self) -> None:
        num_qubits = 4
        qubit_pro1 = 0
        qubit_pro2 = 2
        paulis = PauliMatrices(num_qubits)
        projs = paulis.get_projection_matrices([qubit_pro1, qubit_pro2])
        assert len(projs) == 16

        I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        assert self.in_array(np.kron(np.kron(np.kron(X, I), I), I), projs)
        assert self.in_array(np.kron(np.kron(np.kron(Y, I), I), I), projs)
        assert self.in_array(np.kron(np.kron(np.kron(Z, I), I), I), projs)
        assert self.in_array(np.kron(np.kron(np.kron(I, I), I), I), projs)
        assert self.in_array(np.kron(np.kron(np.kron(X, I), X), I), projs)
        assert self.in_array(np.kron(np.kron(np.kron(Y, I), X), I), projs)
        assert self.in_array(np.kron(np.kron(np.kron(Z, I), X), I), projs)
        assert self.in_array(np.kron(np.kron(np.kron(I, I), X), I), projs)
        assert self.in_array(np.kron(np.kron(np.kron(X, I), Y), I), projs)
        assert self.in_array(np.kron(np.kron(np.kron(Y, I), Y), I), projs)
        assert self.in_array(np.kron(np.kron(np.kron(Z, I), Y), I), projs)
        assert self.in_array(np.kron(np.kron(np.kron(I, I), Y), I), projs)
        assert self.in_array(np.kron(np.kron(np.kron(X, I), Z), I), projs)
        assert self.in_array(np.kron(np.kron(np.kron(Y, I), Z), I), projs)
        assert self.in_array(np.kron(np.kron(np.kron(Z, I), Z), I), projs)
        assert self.in_array(np.kron(np.kron(np.kron(I, I), Z), I), projs)


class TestPauliMatricesDotProduct:

    @pytest.mark.parametrize('invalid_alpha', [1.37, False, ['x']])
    def test_invalid_type(self, invalid_alpha: Any) -> None:
        with pytest.raises(TypeError):
            PauliMatrices(1).dot_product(invalid_alpha)

    @pytest.mark.parametrize('invalid_alpha', [[1.1] * i for i in range(4)])
    def test_invalid_value(self, invalid_alpha: Sequence[float]) -> None:
        with pytest.raises(ValueError):
            PauliMatrices(1).dot_product(invalid_alpha)

    @pytest.mark.parametrize(
        'alpha, prod', [
            ([1, 0, 0, 0], PauliMatrices.I),
            ([0, 1, 0, 0], PauliMatrices.X),
            ([0, 0, 1, 0], PauliMatrices.Y),
            ([0, 0, 0, 1], PauliMatrices.Z),
            ([1, 0, 0, 1], PauliMatrices.I + PauliMatrices.Z),
            ([0, 2, 0, 1], 2 * PauliMatrices.X + PauliMatrices.Z),
            (
                [1, 0, 3, 1],
                PauliMatrices.I
                + 3 * PauliMatrices.Y
                + PauliMatrices.Z,
            ),
            (
                [91.3, 1.3, 1.7, 1],
                91.3 * PauliMatrices.I
                + 1.3 * PauliMatrices.X
                + 1.7 * PauliMatrices.Y
                + PauliMatrices.Z,
            ),
        ],
    )
    def test_size_1(self, alpha: Sequence[float], prod: np.ndarray) -> None:
        assert np.allclose(PauliMatrices(1).dot_product(alpha), prod)

    @pytest.mark.parametrize(
        'alpha, prod', [
            (
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                np.kron(PauliMatrices.I, PauliMatrices.I),
            ),
            (
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                np.kron(PauliMatrices.I, PauliMatrices.X),
            ),
            (
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                np.kron(PauliMatrices.I, PauliMatrices.Y),
            ),
            (
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                np.kron(PauliMatrices.I, PauliMatrices.Z),
            ),
            (
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                np.kron(PauliMatrices.X, PauliMatrices.I),
            ),
            (
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                np.kron(PauliMatrices.X, PauliMatrices.X),
            ),
            (
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                np.kron(PauliMatrices.X, PauliMatrices.Y),
            ),
            (
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                np.kron(PauliMatrices.X, PauliMatrices.Z),
            ),
            (
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                np.kron(PauliMatrices.Y, PauliMatrices.I),
            ),
            (
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                np.kron(PauliMatrices.Y, PauliMatrices.X),
            ),
            (
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                np.kron(PauliMatrices.Y, PauliMatrices.Y),
            ),
            (
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                np.kron(PauliMatrices.Y, PauliMatrices.Z),
            ),
            (
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                np.kron(PauliMatrices.Z, PauliMatrices.I),
            ),
            (
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                np.kron(PauliMatrices.Z, PauliMatrices.X),
            ),
            (
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                np.kron(PauliMatrices.Z, PauliMatrices.Y),
            ),
            (
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                np.kron(PauliMatrices.Z, PauliMatrices.Z),
            ),
            (
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                np.kron(PauliMatrices.I, PauliMatrices.I)
                + np.kron(PauliMatrices.Z, PauliMatrices.Z),
            ),
            (
                [1.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 91.7],
                1.8 * np.kron(PauliMatrices.I, PauliMatrices.I)
                + 91.7 * np.kron(PauliMatrices.Z, PauliMatrices.Z),
            ),
        ],
    )
    def test_size_2(self, alpha: Sequence[float], prod: np.ndarray) -> None:
        assert np.allclose(PauliMatrices(2).dot_product(alpha), prod)


class TestPauliMatricesFromString:

    def in_array(self, needle: Any, haystack: Any) -> bool:
        for elem in haystack:
            if not needle.shape == elem.shape:
                continue
            if np.allclose(elem, needle):
                return True

        return False

    def test_invalid_type(self, not_a_str: Any) -> None:
        with pytest.raises(TypeError):
            paulis = PauliMatrices.from_string(not_a_str)  # noqa

    @pytest.mark.parametrize(
        'invalid_str', [
            'ABC',
            'IXYZA',
            '\t AIXYZ  ,, \n\r\tabc\t',
            'IXYZ+',
            'IXYZ, IXA',
            'WXYZ, XYZ',
        ],
    )
    def test_invalid_value(self, invalid_str: str) -> None:
        with pytest.raises(ValueError):
            paulis = PauliMatrices.from_string(invalid_str)  # noqa

    @pytest.mark.parametrize(
        'pauli_str, pauli_mat', [
            (
                'XYZ',
                np.kron(
                    np.kron(
                        PauliMatrices.X,
                        PauliMatrices.Y,
                    ),
                    PauliMatrices.Z,
                ),
            ),
            (
                'XYX',
                np.kron(
                    np.kron(
                        PauliMatrices.X,
                        PauliMatrices.Y,
                    ),
                    PauliMatrices.X,
                ),
            ),
            (
                'XXI',
                np.kron(
                    np.kron(
                        PauliMatrices.X,
                        PauliMatrices.X,
                    ),
                    PauliMatrices.I,
                ),
            ),
            ('\t XY  ,,\n\r\t\t', np.kron(PauliMatrices.X, PauliMatrices.Y)),
        ],
    )
    def test_single(self, pauli_str: str, pauli_mat: np.ndarray) -> None:
        assert isinstance(PauliMatrices.from_string(pauli_str), np.ndarray)
        assert np.allclose(
            np.array(PauliMatrices.from_string(pauli_str)),
            pauli_mat,
        )

    @pytest.mark.parametrize(
        'pauli_str, pauli_mats', [
            (
                'XYZ, XYZ', [
                    np.kron(
                        np.kron(
                            PauliMatrices.X,
                            PauliMatrices.Y,
                        ),
                        PauliMatrices.Z,
                    ),
                    np.kron(
                        np.kron(
                            PauliMatrices.X,
                            PauliMatrices.Y,
                        ),
                        PauliMatrices.Z,
                    ),
                ],
            ),
            (
                'XYZ, XII', [
                    np.kron(
                        np.kron(
                            PauliMatrices.X,
                            PauliMatrices.Y,
                        ),
                        PauliMatrices.Z,
                    ),
                    np.kron(
                        np.kron(
                            PauliMatrices.X,
                            PauliMatrices.I,
                        ),
                        PauliMatrices.I,
                    ),
                ],
            ),
            (
                'XYZ, XII, IIX', [
                    np.kron(
                        np.kron(
                            PauliMatrices.X,
                            PauliMatrices.Y,
                        ),
                        PauliMatrices.Z,
                    ),
                    np.kron(
                        np.kron(
                            PauliMatrices.X,
                            PauliMatrices.I,
                        ),
                        PauliMatrices.I,
                    ),
                    np.kron(
                        np.kron(
                            PauliMatrices.I,
                            PauliMatrices.I,
                        ),
                        PauliMatrices.X,
                    ),
                ],
            ),
            (
                'XYZ, XII, IIX, \t\n\r  ,, \t\n\rIXXY', [
                    np.kron(
                        np.kron(
                            PauliMatrices.X,
                            PauliMatrices.Y,
                        ),
                        PauliMatrices.Z,
                    ),
                    np.kron(
                        np.kron(
                            PauliMatrices.X,
                            PauliMatrices.I,
                        ),
                        PauliMatrices.I,
                    ),
                    np.kron(
                        np.kron(
                            PauliMatrices.I,
                            PauliMatrices.I,
                        ),
                        PauliMatrices.X,
                    ),
                    np.kron(
                        np.kron(
                            np.kron(
                                PauliMatrices.I,
                                PauliMatrices.X,
                            ),
                            PauliMatrices.X,
                        ),
                        PauliMatrices.Y,
                    ),
                ],
            ),
        ],
    )
    def test_multi(self, pauli_str: str, pauli_mats: np.ndarray) -> None:
        paulis = PauliMatrices.from_string(pauli_str)
        assert isinstance(paulis, list)
        assert all(isinstance(pauli, np.ndarray) for pauli in paulis)
        assert len(paulis) == len(pauli_mats)
        assert all(self.in_array(pauli, pauli_mats) for pauli in paulis)
