"""This module implements many helper functions to check types."""
from __future__ import annotations

import logging
import numbers
from collections.abc import Sequence
from typing import Any

import numpy as np

from bqskit.ir.point import CircuitPoint


_logger = logging.getLogger(__name__)


def is_iterable(test_variable: Any) -> bool:
    """Returns true if test_variable is an iterable object."""
    try:
        iterator = iter(test_variable)  # noqa: F841
        return True
    except TypeError:
        _logger.debug('Invalid iterable.')
        return False


def is_sequence(test_variable: Any) -> bool:
    """Returns true if test_variable is a sequence."""
    if isinstance(test_variable, (Sequence, np.ndarray)):
        return True
    else:
        _logger.debug('Invalid sequence.')
        return False


def is_numeric(test_variable: Any) -> bool:
    """Return true if test_variable is numeric."""
    return (
        isinstance(test_variable, numbers.Number)
        and not isinstance(test_variable, bool)
    )


def is_complex(test_variable: Any) -> bool:
    """Return true if test_variable is complex."""
    return (
        isinstance(test_variable, (complex, np.complex64, np.complex128))
        or np.iscomplex(test_variable)
    )


def is_real_number(test_variable: Any) -> bool:
    """Return true if `test_variable` is a real number."""
    return is_numeric(test_variable) and not is_complex(test_variable)


def is_integer(test_variable: Any) -> bool:
    """Return true if test_variable is an integer."""
    return (
        isinstance(test_variable, (int, np.integer))
        and not isinstance(test_variable, bool)
    )


def is_valid_location(
    location: Sequence[int],
    num_qudits: int | None = None,
) -> bool:
    """
    Determines if the sequence of qudits form a valid location. A valid location
    is a set of qubit indices (integers) that are greater than or equal to zero,
    and if num_qudits is specified, less than num_qudits.

    Args:
        location (Sequence[int]): The location to check.

        num_qudits (int | None): The total number of qudits.
            All qudit indices should be less than this. If None,
            don't check.

    Returns:
        (bool): True if the location is valid.

    """
    if not is_iterable(location):
        return False

    if not all([is_integer(qudit) for qudit in location]):
        _logger.debug('Location is not an iterable of ints.')
        return False

    if len(location) != len(set(location)):
        _logger.debug('Location has duplicates.')
        return False

    if not all([qudit >= 0 for qudit in location]):
        _logger.debug('Location invalid; qudit indices must be nonnegative.')
        return False

    if num_qudits is not None:
        if not all([qudit < num_qudits for qudit in location]):
            _logger.debug('Location has an erroneously large qudit.')
            return False

    return True


def is_valid_radixes(
    radixes: Sequence[int],
    num_qudits: int | None = None,
) -> bool:
    """
    Determines if the sequence of radixes are valid. Radixes must be integers
    greater than or equal to 2. If num_qudits is specified, then the length of
    radixes must be equal to num_qudits.

    Args:
        radixes (Sequence[int]): The radixes to check.

        num_qudits (int | None): The total number of qudits.
            Length of `radixes` should be equal to this. If None,
            don't check.

    Returns:
        (bool): True if the radixes are valid.

    """

    if not is_sequence(radixes):
        return False

    if not all([is_integer(qudit) for qudit in radixes]):
        fail_idx = [is_integer(qudit) for qudit in radixes].index(False)
        _logger.debug(
            'Radixes is not a tuple of ints, got: %s.' % type(
                radixes[fail_idx],
            ),
        )
        return False

    if not all([radix >= 2 for radix in radixes]):
        _logger.debug('Radixes invalid; radix indices must be >= 2.')
        return False

    if num_qudits is not None and len(radixes) != num_qudits:
        _logger.debug('Invalid number of radixes.')
        return False

    return True


def is_valid_coupling_graph(
    coupling_graph: set[tuple[int, int]],
    num_qudits: int | None = None,
) -> bool:
    """
    Checks if the coupling graph is valid.

    Args:
        coupling_graph (set[tuple[int, int]]): The coupling graph
            to check.

        num_qudits (int | None): The total number of qudits. All qudits
            should be less than this. If None, don't check.

    Returns:
        (bool): Valid or not

    """

    if not isinstance(coupling_graph, set):
        _logger.debug('Coupling graph is not a set.')
        return False

    if len(coupling_graph) == 0:
        return True

    if not all(isinstance(pair, tuple) for pair in coupling_graph):
        _logger.debug('Coupling graph is not a sequence of tuples.')
        return False

    if not all([len(pair) == 2 for pair in coupling_graph]):
        _logger.debug('Coupling graph is not a sequence of pairs.')
        return False

    if num_qudits is not None:
        if not (is_integer(num_qudits) and num_qudits > 0):
            _logger.debug('Invalid num_qudits in coupling graph check.')
            return False

        if not all(
            qudit < num_qudits
            for pair in coupling_graph
            for qudit in pair
        ):
            _logger.debug('Coupling graph has invalid qudits.')
            return False

    if not all([
        len(pair) == len(set(pair))
        for pair in coupling_graph
    ]):
        _logger.debug('Coupling graph has an invalid pair.')
        return False

    return True


def is_vector(V: np.ndarray) -> bool:
    """Return true if V is a vector."""

    if not isinstance(V, np.ndarray):
        _logger.debug('V is not an numpy array.')
        return False

    if len(V.shape) != 1:
        _logger.debug('V is not an 1-dimensional array.')
        return False

    if V.dtype.kind not in 'biufc':
        _logger.debug('V is not a numeric array.')
        return False

    return True


def is_matrix(M: np.ndarray) -> bool:
    """Checks if M is a matrix."""

    if not isinstance(M, np.ndarray):
        _logger.debug('M is not an numpy array.')
        return False

    if len(M.shape) != 2:
        _logger.debug('M is not an 2-dimensional array.')
        return False

    if M.dtype.kind not in 'biufc':
        _logger.debug('M is not a numeric array.')
        return False

    return True


def is_square_matrix(M: np.ndarray) -> bool:
    """Checks if M is a square matrix."""

    if not is_matrix(M):
        return False

    if M.shape[0] != M.shape[1]:
        return False

    return True


def is_unitary(U: np.ndarray, tol: float = 1e-8) -> bool:
    """Checks if U is a unitary matrix."""

    if not is_square_matrix(U):
        return False

    X = U @ U.conj().T
    Y = U.conj().T @ U
    I = np.identity(X.shape[0])

    if not np.allclose(X, I, rtol=0, atol=tol):
        if _logger.isEnabledFor(logging.DEBUG):
            norm = np.linalg.norm(X - I)
            _logger.debug('Failed unitary condition, ||UU^d - I|| = %e' % norm)
        return False

    if not np.allclose(Y, I, rtol=0, atol=tol):
        if _logger.isEnabledFor(logging.DEBUG):
            norm = np.linalg.norm(Y - I)
            _logger.debug('Failed unitary condition, ||U^dU - I|| = %e' % norm)
        return False

    return True


def is_hermitian(H: np.ndarray, tol: float = 1e-8) -> bool:
    """Checks if H is a hermitian matrix."""

    if not is_square_matrix(H):
        return False

    if not np.allclose(H, H.conj().T, rtol=0, atol=tol):
        if _logger.isEnabledFor(logging.DEBUG):
            norm = np.linalg.norm(H - H.conj().T)
            _logger.debug(
                'Failed hermitian condition, ||H - H^d|| = %e'
                % norm,
            )
        return False

    return True


def is_skew_hermitian(H: np.ndarray, tol: float = 1e-8) -> bool:
    """Checks if H is a skew hermitian matrix."""

    if not is_square_matrix(H):
        return False

    if not np.allclose(-H, H.conj().T, rtol=0, atol=tol):
        if _logger.isEnabledFor(logging.DEBUG):
            norm = np.linalg.norm(-H - H.conj().T)
            _logger.debug(
                'Failed skew hermitian condition, ||H - H^d|| = %e'
                % norm,
            )
        return False

    return True


def is_permutation(P: np.ndarray, tol: float = 1e-8) -> bool:
    """Checks if P is a permutation matrix."""

    if not is_unitary(P, tol):
        return False

    if not all(s == 1 for s in P.sum(0)):  # type: ignore
        _logger.debug('Not all rows sum to 1.')
        return False

    if not all(s == 1 for s in P.sum(1)):  # type: ignore
        _logger.debug('Not all columns sum to 1.')
        return False

    if not all(e == 1 or e == 0 for row in P for e in row):
        _logger.debug('Not all elements are 0 or 1.')
        return False

    return True


def is_pure_state(V: np.ndarray, tol: float = 1e-8) -> bool:
    """Return true if V is a pure state vector."""
    if not is_vector(V):
        return False

    if not np.allclose(
        np.sum(np.abs([elem for elem in V])),
        1, rtol=0, atol=tol,
    ):
        _logger.debug('Failed pure state criteria.')
        return False

    return True


def is_point(point: Any) -> bool:
    """Return true is point is a CircuitPointLike."""
    if isinstance(point, CircuitPoint):
        return True

    if not isinstance(point, tuple):
        _logger.debug('Point is not a tuple.')
        return False

    if len(point) != 2:
        _logger.debug(
            'Expected point to contain two values, got %d.' % len(point),
        )
        return False

    if not is_integer(point[0]):
        _logger.debug(
            'Expected integer values in point, got %s.' % type(point[0]),
        )
        return False

    if not is_integer(point[1]):
        _logger.debug(
            'Expected integer values in point, got %s.' % type(point[1]),
        )
        return False

    return True
