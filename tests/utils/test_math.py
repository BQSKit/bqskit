"""This module tests the calculus library."""
from __future__ import annotations

import numpy as np
import pytest
import scipy as sp

from bqskit.qis.pauli import PauliMatrices
from bqskit.utils.math import dexpmv


def dexpm_exact(
    M: np.ndarray,
    dM: np.ndarray,
    term_count: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
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
    def test_dexpmv_single(self, alpha: np.ndarray) -> None:
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
    def test_dexpmv_vector(self, alpha: np.ndarray) -> None:
        paulis = PauliMatrices(2)
        H = paulis.dot_product(alpha)

        dFs0 = []
        for p in paulis:
            _, dF = dexpm_exact(H, p)
            dFs0.append(dF)

        dFs0 = np.array(dFs0)

        _, dFs1 = dexpmv(H, paulis.get_numpy())

        assert np.allclose(dFs0, dFs1)

    def test_dexpmv_invalid(self):
        with pytest.raises(Exception):
            dexpmv(0, 0)

        with pytest.raises(Exception):
            dexpmv(0, [1, 0])

        with pytest.raises(Exception):
            dexpmv([1, 0], 0)

        with pytest.raises(Exception):
            dexpmv([1, 0], [1, 0])

        I = np.identity(2)

        with pytest.raises(Exception):
            dexpmv(I, 0)

        with pytest.raises(Exception):
            dexpmv(0, I)

        with pytest.raises(Exception):
            dexpmv(I, [1, 0])

        with pytest.raises(Exception):
            dexpmv([1, 0], I)
