"""This module implements the StateVector class."""
from __future__ import annotations

import logging
from typing import Any
from typing import Iterator
from typing import Sequence
from typing import Union

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

from bqskit.utils.typing import is_valid_radixes
from bqskit.utils.typing import is_vector


_logger = logging.getLogger(__name__)


class StateVector(NDArrayOperatorsMixin):
    """The StateVector class."""

    def __init__(
        self,
        input: StateLike,
        radixes: Sequence[int] = [],
        check_arguments: bool = True,
    ) -> None:
        """
        Constructs a StateVector with the supplied vector.

        Args:
            input (StateLike): The state vector.

            radixes (Sequence[int]): A sequence with its length equal to
                the number of qudits this StateVector represents. Each
                element specifies the base, number of orthogonal states,
                for the corresponding qudit. By default, the constructor
                will attempt to calculate `radixes` from `input`.

        Raises:
            TypeError: If `radixes` is not specified and the constructor
                cannot determine `radixes`.
        """
        # Copy Constructor
        if isinstance(input, StateVector):
            self._vec = input.get_numpy()
            self._radixes = input.radixes
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

    def get_numpy(self) -> np.ndarray:
        return self._vec

    @property
    def shape(self) -> tuple[int, ...]:
        return self._vec.shape

    @property
    def dtype(self) -> np.typing.DTypeLike:
        return self._vec.dtype

    @property
    def num_qudits(self) -> int:
        return len(self.radixes)

    @property
    def dim(self) -> int:
        return self.shape[0]

    @property
    def radixes(self) -> tuple[int, ...]:
        return self._radixes

    def __len__(self) -> int:
        return self.shape[0]

    def __iter__(self) -> Iterator[np.complex128]:
        return self._vec.__iter__()

    def __getitem__(self, index: Any) -> np.complex128 | np.ndarray:
        return self._vec[index]

    def get_probs(self) -> tuple[float, ...]:
        return tuple(np.abs(elem)**2 for elem in self)

    @staticmethod
    def is_pure_state(V: Any, tol: float = 1e-8) -> bool:
        """Return true if V is a pure state vector."""
        if isinstance(V, StateVector):
            return True

        if not np.allclose(np.sum(np.square(np.abs(V))), 1, rtol=0, atol=tol):
            _logger.debug('Failed pure state criteria.')
            return False

        return True

    def __array__(
            self,
            dtype: np.typing.DTypeLike = np.complex128,
    ) -> np.ndarray:
        if dtype != np.complex128:
            raise ValueError('UnitaryMatrix only supports Complex128 dtype.')

        return self._vec

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: np.ndarray,
        **kwargs: Any,
    ) -> StateVector | np.ndarray:
        if method != '__call__':
            return NotImplemented

        non_state_involved = False
        args: list[np.ndarray] = []
        for input in inputs:
            if isinstance(input, StateVector):
                args.append(input.get_numpy())
            else:
                args.append(input)
                non_state_involved = True

        out = ufunc(*args, **kwargs)

        # The results are state vectors
        # if only states are involved
        # and state vectors are closed under the specific operation.
        convert_back = not non_state_involved and (
            ufunc.__name__ == 'conjugate'
        )

        if convert_back:
            return StateVector(out, self.radixes)

        return out

    def __str__(self) -> str:
        return str(self._vec)

    def __repr__(self) -> str:
        return repr(self._vec)


StateLike = Union[StateVector, np.ndarray, Sequence[Any]]
