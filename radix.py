from __future__ import annotations

import logging
from typing import Any
from typing import Sequence
from typing import TYPE_CHECKING
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

if TYPE_CHECKING:
    from typing import TypeGuard

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
        radixes: Union[int, Sequence[int]] = [],
        check_arguments: bool = True,
    ) -> None:
        """
        Constructs a `UnitaryMatrix` from the supplied unitary matrix.

        Args:
            input (UnitaryLike): The unitary matrix input.

            radixes (Union[int, Sequence[int]]): Either a single integer or
                a sequence with its length equal to the number of qudits this
                `UnitaryMatrix` can act on. Each element specifies the base,
                number of orthogonal states, for the corresponding qudit. By
                default, the constructor will attempt to calculate `radixes`
                from `input`.

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
            array([[0.+0.j, 1.+0.j],
                   [1.+0.j, 0.+0.j]])
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

        if isinstance(radixes, int):
            self._radixes = tuple([radixes] * int(np.round(np.log(dim) / np.log(radixes))))
        elif radixes:
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

    def to_special(self) -> UnitaryMatrix:
        """Return a special unitary matrix verson of this one."""
        determinant = np.linalg.det(self)
        dimension = len(self)
        global_phase = np.angle(determinant) / dimension
        global_phase = global_phase % (2 * np.pi / dimension)
        global_phase_factor = np.exp(-1j * global_phase)
        return global_phase_factor * self

    def is_special(self) -> bool:
        """Return true if this unitary is special."""
        return 1 - np.abs(np.linalg.det(self)) < 1e-8

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

    def get_distance_from(self, other: UnitaryLike, degree: int = 2) -> float:
        """
        Return the distance between `self` and `other`.

        The distance is given as:

        .. math::

            \\sqrt[D]{1 - \\frac{|Tr(U_1^\\dagger U_2)|^D}{N^D}}

        where D is the degree, by default is 2.

        Args:
            other (UnitaryLike): The unitary to measure distance from.

            degree (int): The degree of the distance metric.

        Returns:
            float: A value between 1 and 0, where 0 means the two unitaries
            are equal up to global phase and 1 means the two unitaries are
            very unsimilar or far apart.
        """
        other = UnitaryMatrix(other, check_arguments=False)
        num = np.abs(np.trace(self.conj().T @ other))
        dem = self.dim
        frac = min(num / dem, 1)
        dist = np.power(1 - (frac ** degree), 1.0 / degree)
        return dist if dist > 0.0 else 0.0

    def __array__(self, dtype: npt.DTypeLike = None) -> npt.NDArray[np.complex128]:
        """Support for the `np.array` interface."""
        return np.array(self._utry, dtype=dtype)

    def __str__(self) -> str:
        """Return a string representation of the unitary matrix."""
        return str(self._utry)

    def __repr__(self) -> str:
        """Return a string representation of the unitary matrix."""
        return repr(self._utry)

    def __matmul__(self, other: UnitaryLike) -> UnitaryMatrix:
        """Compute the matrix product with `other`."""
        return UnitaryMatrix(
            self._utry @ UnitaryMatrix(other).numpy,
            self.radixes,
            False,
        )

    def __rmatmul__(self, other: UnitaryLike) -> UnitaryMatrix:
        """Compute the reverse matrix product with `other`."""
        return UnitaryMatrix(
            UnitaryMatrix(other).numpy @ self._utry,
            self.radixes,
            False,
        )

    def __getitem__(
        self, key: int | slice
    ) -> complex | npt.NDArray[np.complex128]:
        """Pass through to the unitary."""
        return self._utry[key]

    def __setitem__(
        self, key: int | slice, value: complex | npt.NDArray[np.complex128]
    ) -> None:
        """Pass through to the unitary."""
        self._utry[key] = value

    @staticmethod
    def is_unitary(mat: npt.NDArray[Any]) -> TypeGuard[npt.NDArray[np.complex128]]:
        """Return true if the matrix is unitary."""
        return is_square_matrix(mat) and np.allclose(
            mat @ mat.conj().T,
            np.identity(mat.shape[0], dtype=mat.dtype),
            rtol=1e-8,
            atol=1e-8,
        )
