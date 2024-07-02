"""This module implements numerical functions."""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import scipy as sp

from bqskit.qis.pauli import PauliMatrices
from bqskit.qis.pauliz import PauliZMatrices
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

    norm = np.linalg.norm(M, np.inf)
    e = np.log2(norm) if norm != 0 else -np.inf
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


def pauliz_expansion(H: npt.NDArray[np.complex128]) -> npt.NDArray[np.float64]:
    """
    Computes a Pauli Z expansion of the diagonal hermitian matrix H.

    Args:
        H (np.ndarray): The diagonal hermitian matrix to expand.

    Returns:
        np.ndarray: The coefficients of a Pauli Z expansion for H,
        i.e., X dot Sigma = H where Sigma contains Pauli Z matrices of
        same size of H.

    Note:
        This assumes the input is diagonal. No check is done for hermicity.
        The output is undefined on non-hermitian inputs.
    """
    diag_H = np.diag(np.diag(H))
    if not np.allclose(H, diag_H):
        msg = 'H must be a diagonal matrix.'
        raise ValueError(msg)
    # Change basis of H to Pauli Basis (solve for coefficients -> X)
    n = int(np.log2(len(H)))
    paulizs = PauliZMatrices(n)
    flatten_paulizs = [np.diag(pauli) for pauli in paulizs]
    flatten_H = np.diag(H)
    A = np.stack(flatten_paulizs, axis=-1)
    X = np.real(np.matmul(np.linalg.inv(A), flatten_H))
    return np.array(X)


def compute_su_generators(n: int) -> npt.NDArray[np.complex128]:
    """
    Computes the Lie algebra generators for SU(n).

    Args:
        n (int): dimension of SU(n) algebra

    Returns:
        npt.NDArray[np.complex128]: list of the SU(N) generators,
        note that they are Hermitian, but not neccesarily unitary.

    Raises:
        ValueError: if n<=0

    References:
        https://walterpfeifer.ch/liealgebra/LieAlg_wieBuch4.pdf
    """
    # TODO HermitianMatrix objects

    if n <= 0:
        raise ValueError(f'Expected positive integer for n, got: {n}.')

    elif n == 1:
        return np.array([1], dtype=np.complex128)

    elif n == 2:
        return np.array(
            [
                [[0, 1], [1, 0]],
                [[0, -1j], [1j, 0]],
                [[1, 0], [0, -1]],
            ], dtype=np.complex128,
        )

    else:
        previous_generators = compute_su_generators(n - 1)
        generators = [
            np.pad(previous_generators[i], (0, 1))
            for i in range(len(previous_generators))
        ]
        for i in range(n - 1):
            t = np.zeros((n, n), dtype=np.complex128)
            t[i, n - 1] = 1.0
            t[n - 1, i] = 1.0
            generators.append(t)
            t2 = np.zeros((n, n), dtype=np.complex128)
            t2[i, n - 1] = -1j
            t2[n - 1, i] = 1j
            generators.append(t2)

        t3 = np.eye(n)
        t3[n - 1, n - 1] = -n + 1
        t3 *= np.sqrt(2 / (n * (n - 1)))
        generators.append(t3)
        return np.array(generators, dtype=np.complex128)


def canonical_unitary(
    unitary: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """
    Computes a canonical form for the provided unitary.

    If unitary matrices V, W differ only by a global phase, then
    canonical_unitary(V) == canonical_unitary(W).

    Args:
        unitary (npt.NDArray[np.complex128]): A unitary matrix.

    Returns:
        npt.NDArray[np.complex128]: A unitary matrix.

    References:
        https://arxiv.org/abs/2306.05622
    """
    determinant = np.linalg.det(unitary)
    dimension = len(unitary)
    # Compute special unitary
    global_phase = np.angle(determinant) / dimension
    global_phase = global_phase % (2 * np.pi / dimension)
    global_phase_factor = np.exp(-1j * global_phase)
    special_unitary = global_phase_factor * unitary
    # Standardize speical unitary to account for exp(-i2pi/N) differences
    first_row_mags = np.linalg.norm(special_unitary[0, :], ord=2)
    index = np.argmax(first_row_mags)
    std_phase = np.angle(special_unitary[0, index])
    correction_phase = 0 - std_phase
    std_correction = np.exp(1j * correction_phase)
    return std_correction * special_unitary


def diagonal_distance(unitary: npt.NDArray[np.complex128]) -> float:
    """
    Compute how diagonal a unitary is.

    The diagonal distance measures how closely a unitary can be approx-
    imately inverted by a diagonal unitary. A unitary is approximately
    inverted when the Hilbert-Schmidt distance to the identity is less
    than some threshold.

    Args:
        unitary (np.ndarray): The unitary matrix to check.

    Returns:
        float: The Hilbert-Schmidt distance to the nearest diagonal.
    """
    eps = unitary - np.diag(np.diag(unitary))
    eps2 = eps * eps.conj()
    distance = abs(np.sqrt(eps2.sum(-1).max()))
    return distance
