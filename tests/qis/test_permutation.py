from __future__ import annotations

import numpy as np
import pytest
import scipy as sp
from bqskitrs import swap_bit

from bqskit.qis.permutation import PermutationMatrix


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
            swap_bit('a', 1, 0)

        with pytest.raises(TypeError):
            swap_bit(0, 'a', 0)

        with pytest.raises(TypeError):
            swap_bit(0, 1, 'a')


class TestFromQubitLocation:

    def test_1(self) -> None:
        swap_012 = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])

        perm = PermutationMatrix.from_qubit_location(2, (1, 0))
        assert np.allclose(perm, swap_012)

        perm = PermutationMatrix.from_qubit_location(2, (1,))
        assert np.allclose(perm, swap_012)

        perm = PermutationMatrix.from_qubit_location(2, (0, 1))
        assert np.allclose(perm, np.identity(4))

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
        assert np.allclose(U0, P @ U1 @ P.T)

        U0 = sp.linalg.expm(-1j * IIXX)
        U1 = sp.linalg.expm(-1j * IXIX)
        P = PermutationMatrix.from_qubit_location(4, (0, 2))
        assert np.allclose(U0, P @ U1 @ P.T)

        U0 = sp.linalg.expm(-1j * IXIXIXIX)
        U1 = sp.linalg.expm(-1j * np.kron(XXXX, IIII))
        P = PermutationMatrix.from_qubit_location(8, (1, 3, 5, 7))
        assert np.allclose(U0, P @ U1 @ P.T)

    def test_calc_permutation_matrix_invalid(self) -> None:
        with pytest.raises(TypeError):
            PermutationMatrix.from_qubit_location(4, 'a')
        with pytest.raises(TypeError):
            PermutationMatrix.from_qubit_location(4, ('a'))
        with pytest.raises(TypeError):
            PermutationMatrix.from_qubit_location('a', (0, 1))
