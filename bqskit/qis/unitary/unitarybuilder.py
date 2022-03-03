"""This module implements the UnitaryBuilder class."""
from __future__ import annotations

import logging
from typing import cast
from typing import Sequence

import numpy as np
import numpy.typing as npt

from bqskit.ir.location import CircuitLocation
from bqskit.ir.location import CircuitLocationLike
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitary import Unitary
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_valid_radixes

logger = logging.getLogger(__name__)


class UnitaryBuilder(Unitary):
    """
    An object for fast unitary accumulation using tensor networks.

    A UnitaryBuilder is similar to a StringBuilder in the sense that it is an
    efficient way to string together or accumulate :class:`Unitary` objects.
    This class uses concepts from tensor networks to efficiently multiply
    unitary matrices.
    """

    def __init__(self, num_qudits: int, radixes: Sequence[int] = []) -> None:
        """
        UnitaryBuilder constructor.

        Args:
            num_qudits (int): The number of qudits to build a Unitary for.

            radixes (Sequence[int]): A sequence with its length equal
                to `num_qudits`. Each element specifies the base of a
                qudit. Defaults to qubits.

        Raises:
            ValueError: If `num_qudits` is nonpositive.

            ValueError: If the length of `radixes` is not equal to
                `num_qudits`.

        Examples:
            >>> builder = UnitaryBuilder(4)  # Creates a 4-qubit builder.
        """

        if not is_integer(num_qudits):
            raise TypeError(
                'Expected int for num_qudits, got %s.' %
                type(num_qudits),
            )

        if num_qudits <= 0:
            raise ValueError(
                'Expected positive number for num_qudits, got %d.' %
                num_qudits,
            )

        self._num_qudits = num_qudits
        self._radixes = tuple(radixes if len(radixes) > 0 else [2] * num_qudits)

        if not is_valid_radixes(self.radixes):
            raise TypeError('Invalid qudit radixes.')

        if len(self.radixes) != self.num_qudits:
            raise ValueError(
                'Expected length of radixes to be equal to num_qudits:'
                ' %d != %d' % (len(self.radixes), self.num_qudits),
            )

        self._num_params = 0
        self._dim = int(np.prod(self.radixes))
        self.tensor = np.identity(self.dim, dtype=np.complex128)
        self.tensor = self.tensor.reshape(self.radixes * 2)

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Build the unitary, see :func:`Unitary.get_unitary` for more."""
        utry = self.tensor.reshape((self.dim, self.dim))
        return UnitaryMatrix(utry, self.radixes, False)

    def apply_right(
        self,
        utry: UnitaryMatrix,
        location: CircuitLocationLike,
        inverse: bool = False,
        check_arguments: bool = True,
    ) -> None:
        """
        Apply the specified unitary on the right of this UnitaryBuilder.

        ..
                 .-----.   .------.
              0 -|     |---|      |-
              1 -|     |---| utry |-
                 .     .   '------'
                 .     .
                 .     .
            n-1 -|     |------------
                 '-----'

        Args:
            utry (UnitaryMatrix): The unitary to apply.

            location (CircuitLocationLike): The qudits to apply the unitary on.

            inverse (bool): If true, apply the inverse of the unitary.

            check_arguments (bool): If true, check the inputs for type and
                value errors.

        Raises:
            ValueError: If `utry`'s size does not match the given location.

            ValueError: if `utry`'s radixes does not match the given location.

        Notes:
            - Applying the unitary on the right is equivalent to multiplying
              the unitary on the left of the tensor. The notation comes
              from the quantum circuit perspective.

            - This operation is performed using tensor contraction.
        """

        if check_arguments:
            if not isinstance(utry, UnitaryMatrix):
                raise TypeError('Expected UnitaryMatrix, got %s', type(utry))

            if not CircuitLocation.is_location(location, self.num_qudits):
                raise TypeError('Invalid location.')

            location = CircuitLocation(location)

            if len(location) != utry.num_qudits:
                raise ValueError('Unitary and location size mismatch.')

            for utry_radix, bldr_radix_idx in zip(utry.radixes, location):
                if utry_radix != self.radixes[bldr_radix_idx]:
                    raise ValueError('Unitary and location radix mismatch.')

        left_perm = list(cast(CircuitLocation, location))
        mid_perm = [x for x in range(self.num_qudits) if x not in left_perm]
        right_perm = [x + self.num_qudits for x in range(self.num_qudits)]

        left_dim = int(np.prod([self.radixes[x] for x in left_perm]))

        utry = utry.dagger if inverse else utry

        perm = left_perm + mid_perm + right_perm
        self.tensor = self.tensor.transpose(perm)
        self.tensor = self.tensor.reshape((left_dim, -1))
        self.tensor = utry @ self.tensor

        shape = list(self.radixes) * 2
        shape = [shape[p] for p in perm]
        self.tensor = self.tensor.reshape(shape)
        inv_perm = list(np.argsort(perm))
        self.tensor = self.tensor.transpose(inv_perm)

    def apply_left(
        self,
        utry: UnitaryMatrix,
        location: CircuitLocationLike,
        inverse: bool = False,
        check_arguments: bool = True,
    ) -> None:
        """
        Apply the specified unitary on the left of this UnitaryBuilder.

        ..
                 .------.   .-----.
              0 -|      |---|     |-
              1 -| gate |---|     |-
                 '------'   .     .
                            .     .
                            .     .
            n-1 ------------|     |-
                            '-----'

        Args:
            utry (UnitaryMatrix): The unitary to apply.

            location (CircuitLocationLike): The qudits to apply the unitary on.

            inverse (bool): If true, apply the inverse of the unitary.

            check_arguments (bool): If true, check the inputs for type and
                value errors.

        Raises:
            ValueError: If `utry`'s size does not match the given location.

            ValueError: if `utry`'s radixes does not match the given location.

        Notes:
            - Applying the unitary on the left is equivalent to multiplying
              the unitary on the right of the tensor. The notation comes
              from the quantum circuit perspective.

            - This operation is performed using tensor contraction.
        """

        if check_arguments:
            if not isinstance(utry, UnitaryMatrix):
                raise TypeError('Expected UnitaryMatrix, got %s', type(utry))

            if not CircuitLocation.is_location(location, self.num_qudits):
                raise TypeError('Invalid location.')

            location = CircuitLocation(location)

            if len(location) != utry.num_qudits:
                raise ValueError('Unitary and location size mismatch.')

            for utry_radix, bldr_radix_idx in zip(utry.radixes, location):
                if utry_radix != self.radixes[bldr_radix_idx]:
                    raise ValueError('Unitary and location radix mismatch.')

        location = cast(CircuitLocation, location)
        left_perm = list(range(self.num_qudits))
        mid_perm = [
            x + self.num_qudits
            for x in left_perm
            if x not in location
        ]
        right_perm = [x + self.num_qudits for x in location]

        right_dim = int(
            np.prod([
                self.radixes[x - self.num_qudits]
                for x in right_perm
            ]),
        )

        utry = utry.dagger if inverse else utry

        perm = left_perm + mid_perm + right_perm
        self.tensor = self.tensor.transpose(perm)
        self.tensor = self.tensor.reshape((-1, right_dim))
        self.tensor = self.tensor @ utry

        shape = list(self.radixes) * 2
        shape = [shape[p] for p in perm]
        self.tensor = self.tensor.reshape(shape)
        inv_perm = list(np.argsort(perm))
        self.tensor = self.tensor.transpose(inv_perm)

    def calc_env_matrix(
            self, location: Sequence[int],
    ) -> npt.NDArray[np.complex128]:
        """
        Calculates the environment matrix w.r.t. the specified location.

        Args:
            location (Sequence[int]): Calculate the environment matrix with
                respect to the qudit indices in location.

        Returns:
            np.ndarray: The environmental matrix.
        """

        left_perm = list(range(self.num_qudits))
        left_perm = [x for x in left_perm if x not in location]
        left_perm = left_perm + [x + self.num_qudits for x in left_perm]
        right_perm = list(location) + [x + self.num_qudits for x in location]

        perm = left_perm + right_perm
        a = np.transpose(self.tensor, perm)
        a = np.reshape(
            a, (
                2 ** (self.num_qudits - len(location)),
                2 ** (self.num_qudits - len(location)),
                2 ** len(location),
                2 ** len(location),
            ),
        )
        return np.trace(a)
