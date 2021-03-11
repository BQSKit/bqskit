from __future__ import annotations

import numpy as np
import scipy as sp

import pytest
from bqskit.qis.permutation import PermutationMatrix
from bqskit.qis.permutation import swap
from bqskit.qis.permutation import swap_bit


class TestSwap:

    def test_swap(self) -> None:
        perm = swap(0, 1, 2).list()
        assert perm[0] == 0
        assert perm[1] == 2
        assert perm[2] == 1
        assert perm[3] == 3
        assert len(perm) == 4

    def test_swap_invalid1(self) -> None:
        with pytest.raises(TypeError):
            swap('a', 1, 0)  # type: ignore

        with pytest.raises(TypeError):
            swap(0, 'a', 0)  # type: ignore

        with pytest.raises(TypeError):
            swap(0, 1, 'a')  # type: ignore

    def test_swap_invalid2(self) -> None:
        with pytest.raises(ValueError):
            swap(0, 1, 0)

        with pytest.raises(ValueError):
            swap(1, 2, 1)


class TestSwapBit:

    def test_swap_bit1(self) -> None:
        assert swap_bit(0, 1, 0) == 0
        assert swap_bit(0, 1, 1) == 2
        assert swap_bit(0, 1, 2) == 1
        assert swap_bit(0, 1, 3) == 3
        assert swap_bit(0, 1, 4) == 4
        assert swap_bit(0, 1, 5) == 6
        assert swap_bit(0, 1, 6) == 5
        assert swap_bit(0, 1, 7) == 7
        assert swap_bit(0, 1, 8) == 8

    def test_swap_bit2(self) -> None:
        assert swap_bit(1, 2, 0) == 0
        assert swap_bit(1, 2, 1) == 1
        assert swap_bit(1, 2, 2) == 4
        assert swap_bit(1, 2, 3) == 5
        assert swap_bit(1, 2, 4) == 2
        assert swap_bit(1, 2, 5) == 3
        assert swap_bit(1, 2, 6) == 6
        assert swap_bit(1, 2, 7) == 7
        assert swap_bit(1, 2, 8) == 8

    def test_swap_bit3(self) -> None:
        for i in range(10):
            for j in range(10):
                assert swap_bit(i, j, 2112) == swap_bit(j, i, 2112)

    def test_swap_bit_invalid(self) -> None:
        with pytest.raises(TypeError):
            swap_bit('a', 1, 0)  # type: ignore

        with pytest.raises(TypeError):
            swap_bit(0, 'a', 0)  # type: ignore

        with pytest.raises(TypeError):
            swap_bit(0, 1, 'a')  # type: ignore


class TestFromQubitLocation:

    def test_1(self) -> None:
        swap_012 = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])

        perm = PermutationMatrix.from_qubit_location(2, (1, 0))
        assert np.allclose(perm.get_numpy(), swap_012)

        perm = PermutationMatrix.from_qubit_location(2, (1,))
        assert np.allclose(perm.get_numpy(), swap_012)

        perm = PermutationMatrix.from_qubit_location(2, (0, 1))
        assert np.allclose(perm.get_numpy(), np.identity(4))

    def test_calc_permutation_matrix_big(self) -> None:
        I = np.identity(2, dtype=np.complex128)
        II = np.kron(I, I)
        IIII = np.kron(II, II)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        XX = np.kron(X, X)
        XXI = np.kron(XX, I)
        IXX = np.kron(I, XX)
        IIXX = np.kron(I, IXX)
        IX = np.kron(I, X)
        IXIX = np.kron(IX, IX)
        XXXX = np.kron(XX, XX)
        IXIXIXIX = np.kron(IXIX, IXIX)

        U0 = sp.linalg.expm(-1j * IXX)
        U1 = sp.linalg.expm(-1j * XXI)
        P = PermutationMatrix.from_qubit_location(3, (1, 2))
        assert np.allclose(U0, P.get_numpy() @ U1 @ P.get_numpy().T)

        U0 = sp.linalg.expm(-1j * IIXX)
        U1 = sp.linalg.expm(-1j * IXIX)
        P = PermutationMatrix.from_qubit_location(4, (0, 2))
        assert np.allclose(U0, P.get_numpy() @ U1 @ P.get_numpy().T)

        U0 = sp.linalg.expm(-1j * IXIXIXIX)
        U1 = sp.linalg.expm(-1j * np.kron(XXXX, IIII))
        P = PermutationMatrix.from_qubit_location(8, (1, 3, 5, 7))
        assert np.allclose(U0, P.get_numpy() @ U1 @ P.get_numpy().T)

    def test_calc_permutation_matrix_invalid(self) -> None:
        with pytest.raises(TypeError):
            PermutationMatrix.from_qubit_location(4, 'a')  # type: ignore
        with pytest.raises(TypeError):
            PermutationMatrix.from_qubit_location(4, ('a'))  # type: ignore
        with pytest.raises(TypeError):
            PermutationMatrix.from_qubit_location('a', (0, 1))  # type: ignore
