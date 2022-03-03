"""This module tests the math library."""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import scipy as sp
from scipy.stats import unitary_group

from bqskit.qis.pauli import PauliMatrices
from bqskit.utils.math import dexpmv
from bqskit.utils.math import dot_product
from bqskit.utils.math import pauli_expansion
from bqskit.utils.math import softmax
from bqskit.utils.math import unitary_log_no_i


def dexpm_exact(
    M: npt.NDArray[np.complex128],
    dM: npt.NDArray[np.complex128],
    term_count: int = 100,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """
    Exact matrix exponential derivative calculation.

    Rossmann 2002 Theorem 5 Section 1.2
    """
    adjMp = dM
    total = np.zeros(M.shape, dtype=np.complex128)

    for i in range(term_count):
        total += adjMp
        adjMp = (M @ adjMp) - (adjMp @ M)
        adjMp *= -1
        adjMp /= i + 2

    F = sp.linalg.expm(M)
    dF = F @ total
    return F, dF


class TestDexpmv:

    @pytest.mark.parametrize(
        'alpha', [np.random.random(16) for i in range(100)],
    )
    def test_single(self, alpha: npt.NDArray[np.float64]) -> None:
        paulis = PauliMatrices(2)
        H = paulis.dot_product(alpha)

        for p in paulis:
            F0, dF0 = dexpm_exact(H, p)
            F1, dF1 = dexpmv(H, p)

            assert np.allclose(F0, F1)
            assert np.allclose(dF0, dF1)

    @pytest.mark.parametrize(
        'alpha', [np.random.random(16) for i in range(100)],
    )
    def test_vector(self, alpha: npt.NDArray[np.float64]) -> None:
        paulis = PauliMatrices(2)
        H = paulis.dot_product(alpha)

        dFs0 = []
        for p in paulis:
            _, dF = dexpm_exact(H, p)
            dFs0.append(dF)

        dFs0_np = np.array(dFs0)

        _, dFs1 = dexpmv(H, paulis.numpy)

        assert np.allclose(dFs0_np, dFs1)

    def test_invalid(self) -> None:
        with pytest.raises(Exception):
            dexpmv(0, 0)  # type: ignore

        with pytest.raises(Exception):
            dexpmv(0, [1, 0])  # type: ignore

        with pytest.raises(Exception):
            dexpmv([1, 0], 0)  # type: ignore

        with pytest.raises(Exception):
            dexpmv([1, 0], [1, 0])  # type: ignore

        I = np.identity(2)

        with pytest.raises(Exception):
            dexpmv(I, 0)  # type: ignore

        with pytest.raises(Exception):
            dexpmv(0, I)  # type: ignore

        with pytest.raises(Exception):
            dexpmv(I, [1, 0])  # type: ignore

        with pytest.raises(Exception):
            dexpmv([1, 0], I)  # type: ignore


class TestSoftmax:

    @pytest.mark.parametrize('x', [np.random.random(100) for i in range(100)])
    def test_1(self, x: npt.NDArray[np.float64]) -> None:
        assert np.abs(np.sum(softmax(10 * x)) - 1) < 1e-15

    def test_2(self) -> None:
        x = np.ones(10)
        x[0] = 2
        assert np.argmax(softmax(x)) == 0
        x[0] = 1
        x[5] = 2
        assert np.argmax(softmax(x)) == 5

    @pytest.mark.parametrize('test_variable', ['a', False, True])
    def test_invalid(self, test_variable: Any) -> None:
        with pytest.raises(TypeError):
            softmax(test_variable)


class TestDotProduct:
    def test_valid_1(self) -> None:
        sigma = np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]])
        alpha = np.array([1, 1])

        expected = np.array([[1, 1], [1, 1]])
        assert np.allclose(dot_product(alpha, sigma), expected)

    def test_valid_2(self) -> None:
        sigma = np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]])
        alpha = np.array([0.5, 0.5])

        expected = 0.5 * np.array([[1, 1], [1, 1]])
        assert np.allclose(dot_product(alpha, sigma), expected)

    def test_invalid(self) -> None:
        sigma = 'a'
        alpha = 'b'

        with pytest.raises(TypeError):
            dot_product(alpha, sigma)  # type: ignore


class TestUnitaryLog:

    @pytest.mark.parametrize(
        'reU',
        PauliMatrices(1).paulis
        + PauliMatrices(2).paulis
        + PauliMatrices(3).paulis
        + PauliMatrices(4).paulis
        + [unitary_group.rvs(16) for _ in range(100)],
    )
    def test_valid(self, reU: npt.NDArray[np.complex128]) -> None:
        H = unitary_log_no_i(reU)
        assert np.allclose(H, H.conj().T, rtol=0, atol=1e-15)
        U = sp.linalg.expm(1j * H)
        assert 1 - (np.abs(np.trace(U.conj().T @ reU)) / U.shape[0]) <= 1e-15
        assert np.allclose(
            U.conj().T @ U,
            np.identity(len(U)),
            rtol=0,
            atol=1e-14,
        )
        assert np.allclose(
            U @ U.conj().T,
            np.identity(len(U)),
            rtol=0,
            atol=1e-14,
        )


class TestPauliExpansion:

    @pytest.mark.parametrize(
        'reH',
        PauliMatrices(1).paulis
        + PauliMatrices(2).paulis
        + PauliMatrices(3).paulis
        + PauliMatrices(4).paulis,
    )
    def test_valid(self, reH: npt.NDArray[np.complex128]) -> None:
        alpha = pauli_expansion(reH)
        print(alpha)
        H = PauliMatrices(int(np.log2(reH.shape[0]))).dot_product(alpha)
        assert np.linalg.norm(H - reH) < 1e-16
