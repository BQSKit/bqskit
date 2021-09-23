"""
This module implements the UnitaryMatrix class.

This is a concrete unitary matrix that can be operated on.
"""
from __future__ import annotations

import logging
from typing import Any
from typing import Sequence
from typing import Union

import numpy as np
import scipy as sp
from scipy.stats import unitary_group

from bqskit.qis.state.state import StateLike
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.statemap import StateVectorMap
from bqskit.qis.unitary.unitary import Unitary
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_square_matrix
from bqskit.utils.typing import is_valid_radixes
_logger = logging.getLogger(__name__)


class UnitaryMatrix(np.ndarray, Unitary, StateVectorMap):  # type: ignore
    """The UnitaryMatrix Class."""

    # num_params = 0

    def __new__(
        cls,
        input: UnitaryLike,
        radixes: Sequence[int] = [],
        check_arguments: bool = True,
    ) -> UnitaryMatrix:
        """
        Constructs a UnitaryMatrix with the supplied unitary matrix.

        Args:
            utry (UnitaryLike): The unitary matrix.

            radixes (Sequence[int]): A sequence with its length equal to
                the number of qudits this UnitaryMatrix can act on. Each
                element specifies the base, number of orthogonal states,
                for the corresponding qudit. By default, the constructor
                will attempt to calculate `radixes` from `utry`.

        Raises:
            TypeError: If `radixes` is not specified and the constructor
                cannot determine `radixes`.

        Examples:
            >>> UnitaryMatrix(
            ...     [
            ...         [0, 1],
            ...         [1, 0],
            ...     ],
            ... )  # Creates a single-qubit Pauli X unitary matrix.
        """

        if isinstance(input, UnitaryMatrix):
            return input

        obj = np.asarray(input).view(cls)

        if check_arguments and not is_square_matrix(obj):
            raise TypeError(f'Expected square matrix, got {type(obj)}.')

        if check_arguments and not UnitaryMatrix.is_unitary(obj):
            raise ValueError('Input failed unitary condition.')

        if radixes:
            obj.radixes = tuple(radixes)

        # Check if unitary dimension is a power of two
        elif obj.get_dim() & (obj.get_dim() - 1) == 0:
            obj.radixes = tuple([2] * int(np.round(np.log2(obj.get_dim()))))

        # Check if unitary dimension is a power of three
        elif 3 ** int(np.round(np.log(obj.get_dim()) / np.log(3))) == obj.get_dim():  # noqa
            radixes = [3] * int(np.round(np.log(obj.get_dim()) / np.log(3)))
            obj.radixes = tuple(radixes)

        else:
            raise RuntimeError(
                'Unable to determine radixes'
                ' for UnitaryMatrix with dim %d.' % obj.get_dim(),
            )

        if check_arguments and not is_valid_radixes(obj.radixes):
            raise TypeError('Invalid qudit radixes.')

        if check_arguments and np.prod(obj.radixes) != obj.get_dim():
            raise ValueError('Qudit radixes mismatch with dimension.')

        return obj

    def __array_finalize__(
        self,
        obj: UnitaryMatrix | np.ndarray | None,
    ) -> None:
        if isinstance(obj, UnitaryMatrix):
            self.radixes = getattr(obj, 'radixes', None)
            return

        if obj is None:
            return

        if not is_square_matrix(obj):
            raise TypeError(f'Expected square matrix, got {type(obj)}.')

        if not UnitaryMatrix.is_unitary(obj):
            raise ValueError('Input failed unitary condition.')

        dim = obj.shape[0]

        # Check if unitary dimension is a power of two
        if dim & (dim - 1) == 0:
            self.radixes = tuple([2] * int(np.round(np.log2(dim))))

        # Check if unitary dimension is a power of three
        elif 3 ** int(np.round(np.log(dim) / np.log(3))) == dim:
            self.radixes = tuple([3] * int(np.round(np.log(dim) / np.log(3))))

        else:
            raise RuntimeError(
                'Unable to determine radixes'
                ' for UnitaryMatrix with dim %d.' % dim,
            )

    def get_shape(self) -> tuple[int, int]:
        return self.utry.shape  # type: ignore
    
    def get_numpy(self) -> np.ndarray:
        """For backwards compatibility"""
        return self

    def get_dim(self) -> int:
        return self.shape[0]

    def get_num_params(self) -> int:
        return 0

    def get_radixes(self) -> tuple[int, ...]:
        return self.radixes

    def get_size(self) -> int:
        return len(self.radixes)

    @property
    def dagger(self) -> UnitaryMatrix:
        return self.conj().T

    def get_unitary(self, params: Sequence[float] = []) -> UnitaryMatrix:
        return self

    def get_dagger(self) -> UnitaryMatrix:
        """Returns the conjugate transpose of the unitary matrix."""
        return self.conj().T

    def get_distance_from(self, other: UnitaryLike) -> float:
        """Returns the distance to `other`."""
        other = UnitaryMatrix(other)
        num = np.abs(np.trace(other.conj().T @ self))
        dem = self.dim
        dist = np.sqrt(1 - ((num / dem) ** 2))
        return dist if dist > 0.0 else 0.0

    def get_statevector(self, in_state: StateLike) -> StateVector:
        """Calculate the output state given the `in_state` input state."""
        if not StateVector.is_pure_state(in_state):
            raise TypeError(f'Expected StateVector, got {type(in_state)}.')

        in_state = StateVector(in_state)

        if in_state.get_dim() != self.dim:
            raise ValueError(
                'State unitary dimension mismatch; '
                f'expected {self.dim}, got {in_state.get_dim()}.',
            )

        vec = in_state[:, None]
        out_vec = self @ vec
        return StateVector(out_vec.reshape((-1,)))

    @staticmethod
    def identity(dim: int, radixes: Sequence[int] = []) -> UnitaryMatrix:
        """Returns an identity UnitaryMatrix."""
        if dim <= 0:
            raise ValueError('Invalid dimension for identity matrix.')
        return UnitaryMatrix(np.identity(dim), radixes)

    @staticmethod
    def closest_to(
        M: np.ndarray,
        radixes: Sequence[int] = [],
    ) -> UnitaryMatrix:
        """
        Calculate and return the closest unitary to a given matrix.

        Calculate the unitary matrix U that is closest with respect to the
        operator norm distance to the general matrix M.

        D.M.Reich. “Characterisation and Identification of Unitary Dynamics
        Maps in Terms of Their Action on Density Matrices”

        Args:
            M (np.ndarray): The matrix input.

            radixes (Sequence[int]): The radixes for the Unitary.

        Returns:
            (UnitaryMatrix): The unitary matrix closest to M.
        """

        if not is_square_matrix(M):
            raise TypeError('Expected square matrix.')

        V, _, Wh = sp.linalg.svd(M)
        return UnitaryMatrix(V @ Wh, radixes, False)

    @staticmethod
    def random(size: int, radixes: Sequence[int] = []) -> UnitaryMatrix:
        """
        Sample a random unitary from the haar distribution.

        Args:
            size (np.ndarray): The number of qudits for the matrix. This
                is not the dimension.

            radixes (Sequence[int]): The radixes for the Unitary.

        Returns:
            (UnitaryMatrix): A random unitary matrix.
        """

        if not is_integer(size):
            raise TypeError('Expected int for size, got %s.' % type(size))

        if size <= 0:
            raise ValueError('Expected positive number for size.')

        radixes = tuple(radixes if len(radixes) > 0 else [2] * size)

        if not is_valid_radixes(radixes):
            raise TypeError('Invalid qudit radixes.')

        if len(radixes) != size:
            raise ValueError(
                'Expected length of radixes to be equal to size:'
                ' %d != %d' % (len(radixes), size),
            )

        U = unitary_group.rvs(int(np.prod(radixes)))
        return UnitaryMatrix(U, radixes, False)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Unitary):
            return np.allclose(self, other.get_unitary())

        if isinstance(other, np.ndarray):
            return np.allclose(self, other)

        raise NotImplemented

    def save(self, filename: str) -> None:
        """Saves the unitary to a file."""
        np.savetxt(filename, self)

    @staticmethod
    def from_file(filename: str) -> UnitaryMatrix:
        """Loads a unitary from a file."""
        return UnitaryMatrix(np.loadtxt(filename, dtype=np.complex128))

    @staticmethod
    def is_unitary(U: np.ndarray, tol: float = 1e-8) -> bool:
        """Checks if U is a unitary matrix."""

        if isinstance(U, UnitaryMatrix):
            return True

        if not is_square_matrix(U):
            return False

        X = U @ U.conj().T
        Y = U.conj().T @ U
        I = np.identity(X.shape[0])

        if not np.allclose(X, I, rtol=0, atol=tol):
            if _logger.isEnabledFor(logging.DEBUG):
                norm = np.linalg.norm(X - I)
                _logger.debug(
                    'Failed unitary condition, ||UU^d - I|| = %e' %
                    norm,
                )
            return False

        if not np.allclose(Y, I, rtol=0, atol=tol):
            if _logger.isEnabledFor(logging.DEBUG):
                norm = np.linalg.norm(Y - I)
                _logger.debug(
                    'Failed unitary condition, ||U^dU - I|| = %e' %
                    norm,
                )
            return False

        return True

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: Sequence[np.ndarray],
        out: None | Sequence[np.ndarray] = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray] | np.ndarray | None:
        non_unitary_involved = False
        args = []
        for i, input in enumerate(inputs):
            if isinstance(input, UnitaryMatrix):
                args.append(input.view(np.ndarray))
            else:
                args.append(input)
                non_unitary_involved = True

        outputs: None | Sequence[np.ndarray] | Sequence[None] = out
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, UnitaryMatrix):
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        results = super().__array_ufunc__(  # type: ignore
            ufunc, method, *args, **kwargs,
        )

        if results is NotImplemented:
            return NotImplemented

        if method == 'at':
            return None

        if ufunc.nout == 1:
            results = (results,)

        # The results are unitary
        # if only unitaries are involved
        # and unitaries are closed under the specific operation.
        convert_back = not non_unitary_involved and (
            ufunc.__name__ == 'conjugate'
            or ufunc.__name__ == 'matmul'
            or ufunc.__name__ == 'negative'
            or ufunc.__name__ == 'positive'
        )

        if convert_back:
            results = tuple(
                (
                    np.asarray(result).view(UnitaryMatrix)
                    if output is None else output
                )
                for result, output in zip(results, outputs)
            )
        else:
            results = tuple(
                (result if output is None else output)
                for result, output in zip(results, outputs)
            )

        return results[0] if len(results) == 1 else results


UnitaryLike = Union[UnitaryMatrix, np.ndarray, Sequence[Sequence[Any]]]
