"""This module implements the UnitaryMatrix class."""
from __future__ import annotations

import logging
from typing import Any
from typing import Sequence
from typing import Union

import numpy as np
import numpy.typing as npt
import scipy as sp
from scipy.stats import unitary_group

from bqskit.qis.state.state import StateLike
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.statemap import StateVectorMap
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitary import Unitary
from bqskit.utils.docs import building_docs
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_square_matrix
from bqskit.utils.typing import is_valid_radixes

if not building_docs():
    from numpy.lib.mixins import NDArrayOperatorsMixin
else:
    class NDArrayOperatorsMixin:  # type: ignore
        pass

_logger = logging.getLogger(__name__)


class UnitaryMatrix(Unitary, StateVectorMap, NDArrayOperatorsMixin):
    """A concrete representation of a unitary matrix."""

    def __init__(
        self,
        input: UnitaryLike,
        radixes: Sequence[int] = [],
        check_arguments: bool = True,
    ) -> None:
        """
        Constructs a `UnitaryMatrix` from the supplied unitary matrix.

        Args:
            input (UnitaryLike): The unitary matrix input.

            radixes (Sequence[int]): A sequence with its length equal to
                the number of qudits this `UnitaryMatrix` can act on. Each
                element specifies the base, number of orthogonal states,
                for the corresponding qudit. By default, the constructor
                will attempt to calculate `radixes` from `utry`.

            check_arguments (bool): If true, check arguments for type
                and value errors.

        Raises:
            ValueError: If `input` is not unitary.

            ValueError: If the dimension of `input` does not match the
                expected dimension from `radixes`.

            RuntimeError: If `radixes` is not specified and the
                constructor cannot infer it.

        Examples:
            >>> UnitaryMatrix(
            ...     [
            ...         [0, 1],
            ...         [1, 0],
            ...     ],
            ... )  # Creates a single-qubit Pauli X unitary matrix.
        """

        # Stop any actual logic when building documentation
        if building_docs():
            self._utry: npt.NDArray[np.complex128] = np.array([])
            return

        # Copy constructor
        if isinstance(input, UnitaryMatrix):
            self._utry = input.numpy
            self._radixes = input.radixes
            self._dim = input.dim
            return

        if check_arguments and not is_square_matrix(input):
            raise TypeError(f'Expected square matrix, got {type(input)}.')

        if check_arguments and not UnitaryMatrix.is_unitary(input):
            raise ValueError('Input failed unitary condition.')

        dim = len(input)

        if radixes:
            self._radixes = tuple(radixes)

        # Check if unitary dimension is a power of two
        elif dim & (dim - 1) == 0:
            self._radixes = tuple([2] * int(np.round(np.log2(dim))))

        # Check if unitary dimension is a power of three
        elif 3 ** int(np.round(np.log(dim) / np.log(3))) == dim:  # noqa
            radixes = [3] * int(np.round(np.log(dim) / np.log(3)))
            self._radixes = tuple(radixes)

        else:
            raise RuntimeError(
                'Unable to determine radixes'
                ' for UnitaryMatrix with dim %d.' % dim,
            )

        if check_arguments and not is_valid_radixes(self.radixes):
            raise TypeError('Invalid qudit radixes.')

        if check_arguments and np.prod(self.radixes) != dim:
            raise ValueError('Qudit radixes mismatch with dimension.')

        self._utry = np.array(input, dtype=np.complex128)
        self._dim = dim

    _num_params = 0

    @property
    def numpy(self) -> npt.NDArray[np.complex128]:
        """The NumPy array holding the unitary."""
        return self._utry

    @property
    def shape(self) -> tuple[int, ...]:
        """The two-dimensional square shape of the unitary."""
        return self._utry.shape

    @property
    def dtype(self) -> np.typing.DTypeLike:
        """The NumPy data type of the unitary."""
        return self._utry.dtype

    @property
    def T(self) -> UnitaryMatrix:
        """The transpose of the unitary."""
        return UnitaryMatrix(self._utry.T, self.radixes, False)

    @property
    def dagger(self) -> UnitaryMatrix:
        """The conjugate transpose of the unitary."""
        return self.conj().T

    def __len__(self) -> int:
        """The dimension of the square unitary matrix."""
        return self.shape[0]

    def __iter__(self) -> int:
        """An iterator that iterates through the rows of the matrix."""
        return self._utry.__iter__()

    def conj(self) -> UnitaryMatrix:
        """Return the complex conjugate unitary matrix."""
        return UnitaryMatrix(self._utry.conj(), self.radixes, False)

    def otimes(self, *utrys: UnitaryLike) -> UnitaryMatrix:
        """
        Calculate the tensor or kroneckor product with other unitaries.

        Args:
            utrys (UnitaryLike | Sequence[UnitaryLike]): The unitary or
                unitaries to computer the tensor product with.

        Returns:
            UnitaryMatrix: The resulting unitary matrix.
        """

        utrys = [UnitaryMatrix(u) for u in utrys]

        utry_acm = self.numpy
        radixes_acm = self.radixes
        for utry in utrys:
            utry_acm = np.kron(utry_acm, utry.numpy)
            radixes_acm += utry.radixes

        return UnitaryMatrix(utry_acm, radixes_acm)

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the same object, satisfies the :class:`Unitary` API."""
        return self

    def get_distance_from(self, other: UnitaryLike) -> float:
        """
        Return the distance between `self` and `other`.

        The distance is given as:

        .. math::

            \\sqrt{1 - \\frac{|Tr(U_1^\\dagger U_2)|}{N}^2}

        Args:
            other (UnitaryLike): The unitary to measure distance from.

        Returns:
            float: A value between 1 and 0, where 0 means the two unitaries
            are equal up to global phase and 1 means the two unitaries are
            very unsimilar or far apart.
        """
        other = UnitaryMatrix(other)
        num = np.abs(np.trace(self.conj().T @ other))
        dem = self.dim
        frac = min(num / dem, 1)
        dist = np.sqrt(1 - (frac ** 2))
        return dist if dist > 0.0 else 0.0

    def get_statevector(self, in_state: StateLike) -> StateVector:
        """
        Calculate the output state after applying this unitary to `in_state`.

        Args:
            in_state (StateLike): The input state to apply `self` to.

        Returns:
            StateVector: The output state.

        Raises:
            ValueError: If the state's dimension doesn't match the
                unitary's dimension.
        """
        if not StateVector.is_pure_state(in_state):
            raise TypeError(f'Expected StateVector, got {type(in_state)}.')

        in_state = StateVector(in_state)

        if in_state.dim != self.dim:
            raise ValueError(
                'State unitary dimension mismatch; '
                f'expected {self.dim}, got {in_state.dim}.',
            )

        vec = in_state[:, None]
        out_vec = self @ vec
        return StateVector(out_vec.reshape((-1,)), self.radixes)

    @staticmethod
    def identity(dim: int, radixes: Sequence[int] = []) -> UnitaryMatrix:
        """
        Construct an identity UnitaryMatrix.

        Args:
            dim (int): The dimension of the identity matrix.

            radixes (Sequence[int]): The number of orthogonal states
                for each qudit, defaults to qubits.

        Returns:
            UnitaryMatrix: An identity matrix.

        Raises:
            ValueError: If `dim` is nonpositive.
        """
        if dim <= 0:
            raise ValueError('Invalid dimension for identity matrix.')
        return UnitaryMatrix(np.identity(dim), radixes)

    @staticmethod
    def closest_to(
        M: npt.NDArray[np.complex128],
        radixes: Sequence[int] = [],
    ) -> UnitaryMatrix:
        """
        Calculate and return the closest unitary to a given matrix.

        Calculate the unitary matrix U that is closest with respect to the
        operator norm distance to the general matrix M.

        Args:
            M (np.ndarray): The matrix input.

            radixes (Sequence[int]): The radixes for the Unitary.

        Returns:
            (UnitaryMatrix): The unitary matrix closest to M.

        References:
            D.M.Reich. “Characterisation and Identification of Unitary Dynamics
            Maps in Terms of Their Action on Density Matrices”
        """
        if not is_square_matrix(M):
            raise TypeError('Expected square matrix.')

        V, _, Wh = sp.linalg.svd(M)
        return UnitaryMatrix(V @ Wh, radixes, False)

    @staticmethod
    def random(num_qudits: int, radixes: Sequence[int] = []) -> UnitaryMatrix:
        """
        Sample a random unitary from the haar distribution.

        Args:
            num_qudits (np.ndarray): The number of qudits for the matrix.
                This is not the dimension.

            radixes (Sequence[int]): The radixes for the Unitary.

        Returns:
            (UnitaryMatrix): A random unitary matrix.

        Raises:
            ValueError: If `num_qudits` is nonpositive.

            ValueError: If the length of `radixes` is not equal to
                `num_qudits`.
        """
        if not is_integer(num_qudits):
            raise TypeError(
                f'Expected int for num_qudits, got {type(num_qudits)}.',
            )

        if num_qudits <= 0:
            raise ValueError('Expected positive number for num_qudits.')

        radixes = tuple(radixes if len(radixes) > 0 else [2] * num_qudits)

        if not is_valid_radixes(radixes):
            raise TypeError('Invalid qudit radixes.')

        if len(radixes) != num_qudits:
            raise ValueError(
                'Expected length of radixes to be equal to num_qudits:'
                ' %d != %d' % (len(radixes), num_qudits),
            )

        U = unitary_group.rvs(int(np.prod(radixes)))
        return UnitaryMatrix(U, radixes, False)

    def __eq__(self, other: object) -> bool:
        """Check if `self` is approximately equal to `other`."""
        if isinstance(other, Unitary):
            other_unitary = other.get_unitary()
            if self.shape != other_unitary.shape:
                return False
            return np.allclose(self, other_unitary)

        if isinstance(other, np.ndarray):
            return np.allclose(self, other)

        return NotImplemented

    def save(self, filename: str) -> None:
        """Save the unitary to a file."""
        np.savetxt(filename, self.numpy)

    def __getitem__(
            self, index: Any,
    ) -> np.complex128 | npt.NDArray[np.complex128]:
        """Implements NumPy API for the StateVector class."""
        return self._utry[index]

    @staticmethod
    def from_file(filename: str) -> UnitaryMatrix:
        """Load a unitary from a file."""
        return UnitaryMatrix(np.loadtxt(filename, dtype=np.complex128))

    @staticmethod
    def is_unitary(U: np.typing.ArrayLike, tol: float = 1e-8) -> bool:
        """
        Check if U is a unitary matrix.

        Args:
            U (np.typing.ArrayLike): The matrix to check.

            tol (float): The numerical precision of the check.

        Returns:
            bool: True if U is a unitary.
        """

        if isinstance(U, UnitaryMatrix):
            return True

        if not isinstance(U, np.ndarray):
            U = np.array(U)

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

    def __array__(
        self,
        dtype: np.typing.DTypeLike = np.complex128,
    ) -> npt.NDArray[np.complex128]:
        """Implements NumPy API for the UnitaryMatrix class."""
        if dtype != np.complex128:
            raise ValueError('UnitaryMatrix only supports Complex128 dtype.')

        return self._utry

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: npt.NDArray[Any],
        **kwargs: Any,
    ) -> UnitaryMatrix | npt.NDArray[np.complex128]:
        """Implements NumPy API for the UnitaryMatrix class."""
        if method != '__call__':
            return NotImplemented

        non_unitary_involved = False
        args: list[npt.NDArray[Any]] = []
        for input in inputs:
            if isinstance(input, UnitaryMatrix):
                args.append(input.numpy)
            else:
                args.append(input)
                non_unitary_involved = True

        out = ufunc(*args, **kwargs)

        # The results are unitary
        # if only unitaries are involved
        # and unitaries are closed under the specific operation.
        convert_back = (
            not non_unitary_involved and (
                ufunc.__name__ == 'conjugate'
                or ufunc.__name__ == 'matmul'
                or ufunc.__name__ == 'negative'
                or ufunc.__name__ == 'positive'
            )
            or (
                ufunc.__name__ == 'multiply'
                and all(
                    np.isscalar(input) or isinstance(input, UnitaryMatrix)
                    for input in inputs
                )
                and all(
                    np.abs(np.abs(input) - 1) <= 1e-14
                    for input in inputs if np.isscalar(input)
                )
            )
        )

        if convert_back:
            return UnitaryMatrix(out, self.radixes)

        return out

    def __str__(self) -> str:
        """Return the string representation of the unitary."""
        return str(self._utry)

    def __repr__(self) -> str:
        """Return the repr representation of the unitary."""
        return repr(self._utry)

    def __hash__(self) -> int:
        """Return the hash of the unitary."""
        return hash((self._utry[0][0], self._utry[-1][-1], self.shape))


UnitaryLike = Union[UnitaryMatrix, np.ndarray, Sequence[Sequence[Any]]]
