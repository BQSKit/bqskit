"""This module implements the StateVector class."""
from __future__ import annotations

import logging
from typing import Any
from typing import Iterator
from typing import Sequence
from typing import TYPE_CHECKING
from typing import Union

import numpy as np
import numpy.typing as npt
from numpy.lib.mixins import NDArrayOperatorsMixin
from scipy.stats import unitary_group

from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_valid_radixes
from bqskit.utils.typing import is_vector


if TYPE_CHECKING:
    from typing import TypeGuard


_logger = logging.getLogger(__name__)


class StateVector(NDArrayOperatorsMixin):
    """A vector representing a pure quantum state."""

    def __init__(
        self,
        input: StateLike,
        radixes: Sequence[int] = [],
        check_arguments: bool = True,
    ) -> None:
        """
        Constructs a `StateVector` from the supplied vector.

        Args:
            input (StateLike): The state vector input.

            radixes (Sequence[int]): A sequence with its length equal to
                the number of qudits this `StateVector` represents. Each
                element specifies the base, number of orthogonal states,
                for the corresponding qudit. By default, the constructor
                will attempt to calculate `radixes` from `input`.

            check_arguments (bool): If true, check arguments for type
                and value errors.

        Raises:
            ValueError: If `input` is not a pure quantum state.

            ValueError: If the dimension of `input` does not match the
                expected dimension from `radixes`.

            RuntimeError: If `radixes` is not specified and the
                constructor cannot infer it.
        """
        # Copy Constructor
        if isinstance(input, StateVector):
            self._vec = input.numpy
            self._radixes = input.radixes
            self._dim = input.dim
            return

        if check_arguments and not is_vector(input):
            raise TypeError(f'Expected vector, got {type(input)}.')

        if check_arguments and not StateVector.is_pure_state(input):
            raise ValueError('Input failed state vector condition.')

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
                ' for StateVector with dim %d.' % dim,
            )

        if check_arguments and not is_valid_radixes(self.radixes):
            raise TypeError('Invalid qudit radixes.')

        if check_arguments and np.prod(self.radixes) != dim:
            raise ValueError('Qudit radixes mismatch with dimension.')

        self._vec = np.array(input, dtype=np.complex128)
        self._dim = dim

    @property
    def numpy(self) -> npt.NDArray[np.complex128]:
        """The NumPy array holding the vector."""
        return self._vec

    @property
    def shape(self) -> tuple[int, ...]:
        """The one-dimensional shape of the vector."""
        return self._vec.shape

    @property
    def dtype(self) -> np.typing.DTypeLike:
        """The NumPy data type of the vector."""
        return self._vec.dtype

    @property
    def num_qudits(self) -> int:
        """The number of qudits in the state."""
        return len(self.radixes)

    @property
    def dim(self) -> int:
        """The vector dimension for this state."""
        return self._dim

    @property
    def radixes(self) -> tuple[int, ...]:
        """The number of orthogonal states for each qudit."""
        return self._radixes

    def __len__(self) -> int:
        """The dimension of the state vector."""
        return self.shape[0]

    def __iter__(self) -> Iterator[np.complex128]:
        """An iterator that iterates through the elements of the vector."""
        return self._vec.__iter__()

    def __getitem__(
            self, index: Any,
    ) -> np.complex128 | npt.NDArray[np.complex128]:
        """Implements NumPy API for the StateVector class."""
        return self._vec[index]

    def get_probs(self) -> tuple[float, ...]:
        """Return the probabilities for each classical outcome."""
        return tuple(np.abs(elem)**2 for elem in self)

    def is_qubit_only(self) -> bool:
        """Return true if this unitary can only act on qubits."""
        return all([radix == 2 for radix in self.radixes])

    def is_qutrit_only(self) -> bool:
        """Return true if this unitary can only act on qutrits."""
        return all([radix == 3 for radix in self.radixes])

    def is_qudit_only(self, radix: int) -> bool:
        """
        Return true if this unitary can only act on `radix`-qudits.

        Args:
            radix (int): Check all qudits have this many orthogonal
                states.
        """
        return all([r == radix for r in self.radixes])

    @staticmethod
    def is_pure_state(V: Any, tol: float = 1e-8) -> TypeGuard[StateLike]:
        """
        Check if V is a pure state vector.

        Args:
            V (Any): The vector to check.

            tol (float): The numerical precision of the check.

        Returns:
            bool: True if V is a pure quantum state vector.
        """
        if isinstance(V, StateVector):
            return True

        if not np.allclose(np.sum(np.square(np.abs(V))), 1, rtol=0, atol=tol):
            _logger.debug('Failed pure state criteria.')
            return False

        return True

    @staticmethod
    def random(num_qudits: int, radixes: Sequence[int] = []) -> StateVector:
        """
        Sample a random pure state.

        Args:
            num_qudits (np.ndarray): The number of qudits in the state.
                This is not the dimension.

            radixes (Sequence[int]): The radixes for the StateVector.

        Returns:
            StateVector: A random pue quantum state.

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
        return StateVector(U[:, 0], radixes)

    def __eq__(self, other: object) -> bool:
        """Check if `self` is approximately equal to `other`."""
        if isinstance(other, StateVector):
            return np.allclose(self.numpy, other.numpy)

        if isinstance(other, np.ndarray):
            return np.allclose(self.numpy, other)

        return NotImplemented

    def __array__(
        self,
        dtype: np.typing.DTypeLike = np.complex128,
    ) -> npt.NDArray[np.complex128]:
        """Implements NumPy API for the StateVector class."""
        if dtype != np.complex128:
            raise ValueError('StateVector only supports Complex128 dtype.')

        return self._vec

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: npt.NDArray[Any],
        **kwargs: Any,
    ) -> StateVector | npt.NDArray[np.complex128]:
        """Implements NumPy API for the StateVector class."""
        if method != '__call__':
            return NotImplemented

        non_state_involved = False
        args: list[npt.NDArray[Any]] = []
        for input in inputs:
            if isinstance(input, StateVector):
                args.append(input.numpy)
            else:
                args.append(input)
                non_state_involved = True

        out = ufunc(*args, **kwargs)

        # The results are state vectors
        # if only states are involved
        # and state vectors are closed under the specific operation.
        convert_back = (
            not non_state_involved and ufunc.__name__ == 'conjugate'
            or (
                ufunc.__name__ == 'multiply'
                and all(
                    np.isscalar(input) or isinstance(input, StateVector)
                    for input in inputs
                )
                and all(
                    np.abs(np.abs(input) - 1) <= 1e-14
                    for input in inputs if np.isscalar(input)
                )
            )
        )

        if convert_back:
            return StateVector(out, self.radixes)

        return out

    def __str__(self) -> str:
        """Return the string representation of the vector."""
        return str(self._vec)

    def __repr__(self) -> str:
        """Return the repr representation of the vector."""
        return repr(self._vec)


StateLike = Union[StateVector, np.ndarray, Sequence[Any]]
