"""
This module implements the UnitaryMatrix class.

This is a concrete unitary matrix that can be operated on.
"""
from __future__ import annotations

from typing import Any
from typing import Sequence
from typing import Union

import numpy as np
import scipy as sp

from bqskit.qis.unitary.unitary import Unitary
from bqskit.utils.typing import is_square_matrix
from bqskit.utils.typing import is_unitary
from bqskit.utils.typing import is_valid_radixes


class UnitaryMatrix(Unitary):
    """The UnitaryMatrix Class."""

    def __init__(
        self,
        utry: UnitaryLike,
        radixes: Sequence[int] = [],
        check_arguments: bool = True,
    ) -> None:
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

        # Copy Constructor
        if isinstance(utry, UnitaryMatrix):
            self.utry = utry.get_numpy()
            self.dim = utry.get_dim()
            self.num_params = utry.get_num_params()
            self.radixes = utry.get_radixes()
            self.size = utry.get_size()
            return

        np_utry = np.array(utry, dtype=np.complex128)

        if check_arguments and not is_unitary(np_utry):
            raise TypeError('Expected unitary matrix.')

        self.utry = np_utry
        self.dim = self.utry.shape[0]
        self.num_params = 0

        if radixes:
            self.radixes = tuple(radixes)

        # Check if unitary dimension is a power of two
        elif self.dim & (self.dim - 1) == 0:
            self.radixes = tuple([2] * int(np.round(np.log2(self.dim))))

        # Check if unitary dimension is a power of three
        elif 3 ** int(np.round(np.log(self.dim) / np.log(3))) == self.dim:
            radixes = [3] * int(np.round(np.log(self.dim) / np.log(3)))
            self.radixes = tuple(radixes)

        else:
            raise TypeError(
                'Unable to determine radixes'
                ' for UnitaryMatrix with dim %d.' % self.dim,
            )

        if check_arguments and not is_valid_radixes(self.radixes):
            raise TypeError('Invalid qudit radixes.')

        if check_arguments and np.prod(self.radixes) != self.dim:
            raise ValueError('Qudit radixes mismatch with dimension.')

        self.size = len(self.radixes)

    def get_numpy(self) -> np.ndarray:
        return self.utry

    def get_shape(self) -> tuple[int, int]:
        return self.utry.shape  # type: ignore

    def get_unitary(self, params: Sequence[float] = []) -> UnitaryMatrix:
        return self

    def get_dagger(self) -> UnitaryMatrix:
        """Returns the conjugate transpose of the unitary matrix."""
        return UnitaryMatrix(self.utry.conj().T, self.get_radixes(), False)

    def get_distance_from(self, other: UnitaryMatrix) -> float:
        """Returns the distance to `other`."""
        num = np.abs(np.trace(other.get_numpy().conj().T @ self.get_numpy()))
        dem = self.get_dim()
        dist = 1 - (num / dem)
        return dist if dist > 0.0 else 0.0

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

    def __matmul__(self, rhs: object) -> UnitaryMatrix:
        if isinstance(rhs, UnitaryMatrix):
            rhs = rhs.get_numpy()
        res: np.ndarray = self.get_numpy() @ rhs  # type: ignore
        return UnitaryMatrix(res, self.get_radixes())

    def save(self, filename: str) -> None:
        """Saves the unitary to a file."""
        np.savetxt(filename, self.utry)

    @staticmethod
    def from_file(filename: str) -> UnitaryMatrix:
        """Loads a unitary from a file."""
        return UnitaryMatrix(np.loadtxt(filename, dtype=np.complex128))


UnitaryLike = Union[UnitaryMatrix, np.ndarray, Sequence[Sequence[Any]]]
