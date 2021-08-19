"""This module implements the StateVector class."""
from __future__ import annotations

import logging
from typing import Any
from typing import Sequence
from typing import Union

import numpy as np

from bqskit.utils.typing import is_valid_radixes
from bqskit.utils.typing import is_vector


_logger = logging.getLogger(__name__)


class StateVector(np.ndarray):
    """The StateVector class."""

    def __new__(
        cls,
        input: StateLike,
        radixes: Sequence[int] = [],
        check_arguments: bool = True,
    ) -> StateVector:
        """
        Constructs a StateVector with the supplied vector.

        Args:
            vec (StateLike): The state vector.

            radixes (Sequence[int]): A sequence with its length equal to
                the number of qudits this StateVector represents. Each
                element specifies the base, number of orthogonal states,
                for the corresponding qudit. By default, the constructor
                will attempt to calculate `radixes` from `vec`.

        Raises:
            TypeError: If `radixes` is not specified and the constructor
                cannot determine `radixes`.
        """
        # Copy Constructor
        if isinstance(input, StateVector):
            return input

        obj = np.asarray(input).view(cls)

        if check_arguments and not is_vector(obj):
            raise TypeError(f'Expected vector, got {type(obj)}.')

        if check_arguments and not StateVector.is_pure_state(obj):
            raise ValueError('Input failed state vector condition.')

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

    def __array_finalize__(self, obj: StateVector | np.ndarray | None) -> None:
        if isinstance(obj, StateVector):
            self.radixes = getattr(obj, 'radixes', None)
            return

        if obj is None:
            return

        if not is_vector(obj):
            raise TypeError(f'Expected vector, got {type(obj)}.')

        if not StateVector.is_pure_state(obj):
            raise ValueError('Input failed state vector condition.')

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

    def get_size(self) -> int:
        return len(self.radixes)

    def get_dim(self) -> int:
        return self.shape[0]

    def get_radixes(self) -> tuple[int, ...]:
        return self.radixes

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

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: Sequence[np.ndarray],
        out: None | Sequence[np.ndarray] = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray] | np.ndarray | None:
        non_state_involved = False
        args = []
        for i, input in enumerate(inputs):
            if isinstance(input, StateVector):
                args.append(input.view(np.ndarray))
            else:
                args.append(input)
                non_state_involved = True

        outputs: None | Sequence[np.ndarray] | Sequence[None] = out
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, StateVector):
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

        # The results are state vectors
        # if only unitaries are involved
        # and unitaries are closed under the specific operation.
        convert_back = not non_state_involved and (
            ufunc.__name__ == 'conjugate'
        )

        if convert_back:
            results = tuple(
                (
                    np.asarray(result).view(StateVector)
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


StateLike = Union[StateVector, np.ndarray, Sequence[Any]]
