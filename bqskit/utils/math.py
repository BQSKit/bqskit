"""This module implements some numerical functions."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from bqskit.utils.typing import is_numeric
from bqskit.utils.typing import is_sequence


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


def dot_product(alpha: Sequence[float], sigma: np.ndarray) -> np.ndarray:
    """
    Computes the standard dot product of `alpha` with `sigma`.

    Args:
        alpha (Sequence[float]): The alpha coefficients.

        sigma (np.ndarray): Sequence of matrices or vector of numbers.

    Returns:
        (np.ndarray): Sum of element-wise multiplication of `alpha`
            and `sigma`.

    Raises:
        ValueError: If `alpha` and `sigma` are incompatible.

    """

    if not is_sequence(alpha) or not all(is_numeric(a) for a in alpha):
        raise TypeError(
            'Expected a sequence of numbers, got %s.' % type(alpha),
        )

    if len(alpha) != len(sigma):
        raise ValueError(
            'Incorrect number of alpha values, expected %d, got %d.'
            % (len(sigma), len(alpha)),
        )

    return np.array(np.sum([a * s for a, s in zip(alpha, sigma)], 0))
