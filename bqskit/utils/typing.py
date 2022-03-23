"""This module implements many helper functions to check types."""
from __future__ import annotations

import logging
import numbers
from collections.abc import Sequence
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Sized

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeGuard


_logger = logging.getLogger(__name__)


def is_iterable(x: Any) -> TypeGuard[Iterable[Any]]:
    """Return true if x is an iterable object."""
    try:
        iterator = iter(x)  # noqa: F841
        return True
    except TypeError:
        return False


def is_sized(x: Any) -> TypeGuard[Sized]:
    """Return true if x is a sized object."""
    try:
        length = len(x)  # noqa: F841
        return True
    except TypeError:
        return False


def is_sequence(x: Any) -> TypeGuard[Sequence[Any]]:
    """Return true if x is a sequence."""
    return isinstance(x, (Sequence, np.ndarray))


def is_mapping(x: Any) -> TypeGuard[Mapping[Any, Any]]:
    """Return true if x is a mapping."""
    return isinstance(x, Mapping)


def is_numeric(x: Any) -> TypeGuard[numbers.Number]:
    """Return true if x is numeric."""
    return isinstance(x, numbers.Number) and not isinstance(x, bool)


def is_complex(x: Any) -> TypeGuard[complex]:
    """Return true if x is complex."""
    return isinstance(x, numbers.Complex)


def is_real_number(x: Any) -> TypeGuard[float]:
    """Return true if `x` is a real number."""
    return isinstance(x, numbers.Real)


def is_integer(x: Any) -> TypeGuard[int]:
    """Return true if x is an integer."""
    return isinstance(x, numbers.Integral) and not isinstance(x, bool)


def is_bool(x: Any) -> TypeGuard[bool]:
    """Return true if x is a boolean value."""
    return isinstance(x, (bool, np.bool_))


def is_sequence_of_int(x: Any) -> TypeGuard[Sequence[int]]:
    """Return true if x is a sequence of integers."""
    return is_sequence(x) and all([is_integer(xi) for xi in x])


def is_valid_radixes(
    radixes: Sequence[int],
    num_qudits: int | None = None,
) -> bool:
    """
    Determine if the sequence of radixes is valid. Radixes must be integers
    greater than or equal to 2. If num_qudits is specified, then the length of
    radixes must be equal to num_qudits.

    Args:
        radixes (Sequence[int]): The radixes to check.

        num_qudits (int | None): The total number of qudits.
            Length of `radixes` should be equal to this. If None,
            don't check.

    Returns:
        bool: True if the radixes are valid.
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


def is_vector(V: np.typing.ArrayLike) -> bool:
    """Return true if V is a vector."""

    if not isinstance(V, np.ndarray):
        V = np.array(V)

    if len(V.shape) != 1:
        _logger.debug('V is not an 1-dimensional array.')
        return False

    if V.dtype.kind not in 'biufc':
        _logger.debug('V is not a numeric array.')
        return False

    return True


def is_matrix(M: np.typing.ArrayLike) -> bool:
    """Return true if M is a matrix."""

    if not isinstance(M, np.ndarray):
        M = np.array(M)

    if len(M.shape) != 2:
        _logger.debug('M is not an 2-dimensional array.')
        return False

    if M.dtype.kind not in 'biufc':
        _logger.debug('M is not a numeric array.')
        return False

    return True


def is_square_matrix(M: np.typing.ArrayLike) -> bool:
    """Return true if M is a square matrix."""

    if not isinstance(M, np.ndarray):
        M = np.array(M)

    if not is_matrix(M):
        return False

    if M.shape[0] != M.shape[1]:
        return False

    return True


def is_hermitian(H: npt.NDArray[np.complex128], tol: float = 1e-8) -> bool:
    """Return true if H is a hermitian matrix."""

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


def is_skew_hermitian(H: npt.NDArray[np.complex128], tol: float = 1e-8) -> bool:
    """Return true if H is a skew hermitian matrix."""

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
