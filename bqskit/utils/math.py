"""This module implements some numerical functions."""
from __future__ import annotations

import numpy as np
import scipy as sp

from bqskit.qis.pauli import PauliMatrices
from bqskit.utils.typing import is_hermitian
from bqskit.utils.typing import is_unitary


def dexpmv(M: np.ndarray, dM: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Matrix exponential F = e^M and its derivative dF.

    User must provide M and its derivative dM. If the argument dM is a
    vector of partials then dF will be the respective partial vector.
    This is done using a Pade Approximat with scaling and squaring.

    Brančík, Lubomír. "Matlab programs for matrix exponential function
    derivative evaluation." Proc. of Technical Computing Prague 2008
    (2008): 17-24.

    Args:
        M (np.ndarray): Matrix to exponentiate.

        dM (np.ndarray): Derivative(s) of M.

    Returns:
        F (np.ndarray): Exponentiated matrix, i.e. e^M.

        dF (np.ndarray): Derivative(s) of F.
    """

    e = np.log2(np.linalg.norm(M, np.inf))
    r = int(max(0, e + 1))
    M = M / (2 ** r)
    dM = dM / (2 ** r)
    X = M
    Y = dM
    c = 0.5
    F = np.identity(M.shape[0]) + c * M
    D = np.identity(M.shape[0]) - c * M
    dF = c * dM
    dD = -c * dM
    q = 6
    p = True
    for k in range(2, q + 1):
        c = c * (q - k + 1) / (k * (2 * q - k + 1))
        Y = dM @ X + M @ Y
        X = M @ X
        cX = c * X
        cY = c * Y
        F = F + cX
        dF = dF + cY
        if p:
            D = D + cX
            dD = dD + cY
        else:
            D = D - cX
            dD = dD - cY
        p = not p
    Dinv = np.linalg.inv(D)
    F = Dinv @ F
    dF = Dinv @ (dF - dD @ F)

    for k in range(1, r + 1):
        dF = dF @ F + F @ dF
        F = F @ F

    return F, dF


def softmax(x: np.ndarray, beta: int = 20) -> np.ndarray:
    """
    Computes the softmax of vector x.

    Args:
        x (np.ndarray): Input vector to softmax.

        beta (int): Beta coefficient to scale steepness of softmax.

    Returns:
        (np.ndarray): Output vector of softmax.
    """

    shiftx = beta * (x - np.max(x))
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def dot_product(alpha: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Computes the standard dot product of `alpha` with `sigma`.

    Args:
        alpha (np.ndarray): The alpha vector.

        sigma (np.ndarray): The sigma vector.

    Returns:
        (np.ndarray): Sum of element-wise multiplication of `alpha`
            and `sigma`.
    """

    return np.array(np.sum([a * s for a, s in zip(alpha, sigma)], 0))


def unitary_log_no_i(U: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """
    Solves for H in U = e^{iH}

    Args:
        U (np.ndarray): The unitary to decompose.

    Returns:
        H (np.ndarray): e^{iH} = U.
    """

    if not is_unitary(U, tol):
        raise TypeError('Expected U to be unitary, got %s.' % type(U))

    T, Z = sp.linalg.schur(U)
    T = np.diag(T)
    D = T / np.abs(T)
    D = np.diag(np.log(D))
    H0 = -1j * (Z @ D @ Z.conj().T)
    return 0.5 * H0 + 0.5 * H0.conj().T


def pauli_expansion(H: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """
    Computes a Pauli expansion of the hermitian matrix H.

    Args:
        H (np.ndarray): The hermitian matrix to expand.

    Returns:
        (np.ndarray): The coefficients of a Pauli expansion for H,
            i.e., X dot Sigma = H where Sigma is Pauli matrices of
            same size of H.
    """

    if not is_hermitian(H, tol):
        raise TypeError('Expected H to be hermitian, got %s.' % type(H))

    # Change basis of H to Pauli Basis (solve for coefficients -> X)
    n = int(np.log2(len(H)))
    paulis = PauliMatrices(n)
    flatten_paulis = [np.reshape(pauli, 4 ** n) for pauli in paulis]
    flatten_H = np.reshape(H, 4 ** n)
    A = np.stack(flatten_paulis, axis=-1)
    X = np.real(np.matmul(np.linalg.inv(A), flatten_H))
    return np.array(X)
