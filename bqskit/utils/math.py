"""This module implements numerical functions."""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import scipy as sp

from bqskit.qis.pauli import PauliMatrices
from bqskit.qis.unitary.unitary import RealVector


def dexpmv(
    M: npt.NDArray[np.complex128], dM: npt.NDArray[np.complex128],
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """
    Compute the Matrix exponential F = e^M and its derivative dF.

    User must provide M and its derivative dM. If the argument dM is a
    vector of partials then dF will be the respective partial vector.
    This is done using a Pade Approximat with scaling and squaring.

    Args:
        M (np.ndarray): Matrix to exponentiate.

        dM (np.ndarray): Derivative(s) of M.

    Returns:
        tuple: Tuple containing
            - F (np.ndarray): Exponentiated matrix, i.e. e^M.

            - dF (np.ndarray): Derivative(s) of F.

    References:
        Brančík, Lubomír. "Matlab programs for matrix exponential function
        derivative evaluation." Proc. of Technical Computing Prague 2008
        (2008): 17-24.
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


def softmax(
    x: npt.NDArray[np.float64],
    beta: int = 20,
) -> npt.NDArray[np.float64]:
    """
    Computes the softmax of vector x.

    Args:
        x (np.ndarray): Input vector to softmax.

        beta (int): Beta coefficient to scale steepness of softmax.

    Returns:
        np.ndarray: Output vector of softmax.
    """

    shiftx = beta * (x - np.max(x))
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def dot_product(alpha: RealVector, sigma: RealVector) -> npt.NDArray[Any]:
    """
    Computes the standard dot product of `alpha` with `sigma`.

    Args:
        alpha (RealVector): The alpha vector.

        sigma (RealVector): The sigma vector.

    Returns:
        np.ndarray: Sum of element-wise multiplication of `alpha`
        and `sigma`.
    """

    return np.array(np.sum([a * s for a, s in zip(alpha, sigma)], 0))


def unitary_log_no_i(
        U: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """
    Solves for H in U = e^{iH}

    Args:
        U (np.ndarray): The unitary to decompose.

    Returns:
        np.ndarray: H in e^{iH} = U.

    Note:
        This assumes the input is unitary but does not check. The output
        is undefined on non-unitary inputs.
    """

    T, Z = sp.linalg.schur(U)
    T = np.diag(T)
    D = T / np.abs(T)
    D = np.diag(np.log(D))
    H0 = -1j * (Z @ D @ Z.conj().T)
    return 0.5 * H0 + 0.5 * H0.conj().T


def pauli_expansion(H: npt.NDArray[np.complex128]) -> npt.NDArray[np.float64]:
    """
    Computes a Pauli expansion of the hermitian matrix H.

    Args:
        H (np.ndarray): The hermitian matrix to expand.

    Returns:
        np.ndarray: The coefficients of a Pauli expansion for H,
        i.e., X dot Sigma = H where Sigma is Pauli matrices of
        same size of H.

    Note:
        This assumes the input is hermitian but does not check. The
        output is undefined on non-hermitian inputs.
    """

    # Change basis of H to Pauli Basis (solve for coefficients -> X)
    n = int(np.log2(len(H)))
    paulis = PauliMatrices(n)
    flatten_paulis = [np.reshape(pauli, 4 ** n) for pauli in paulis]
    flatten_H = np.reshape(H, 4 ** n)
    A = np.stack(flatten_paulis, axis=-1)
    X = np.real(np.matmul(np.linalg.inv(A), flatten_H))
    return np.array(X)
