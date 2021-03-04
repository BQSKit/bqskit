"""
This module implements the UnitaryMatrix class.

This is a concrete unitary matrix that can be operated on.
"""
from __future__ import annotations

from typing import Any
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import scipy as sp

from bqskit.qis.unitary import Unitary
from bqskit.utils.typing import is_square_matrix, is_valid_radixes
from bqskit.utils.typing import is_unitary


class UnitaryMatrix(Unitary):
    """The UnitaryMatrix Class."""

    def __init__(self, utry: UnitaryLike, radixes: Sequence[int] = []) -> None:
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

        np_utry = np.array(utry, dtype=np.complex128)

        if not is_unitary(np_utry):
            raise TypeError('Expected unitary matrix.')

        self.utry = np_utry
        self.dim = self.utry.shape[0]
        self.num_params = 0

        if radixes:
            self.radixes = radixes

        # Check if unitary dimension is a power of two
        elif self.dim & (self.dim - 1) == 0:
            self.radixes = [2] * int(np.round(np.log2(self.dim)))
        
        # Check if unitary dimension is a power of three
        elif 3 ** int(np.round(np.log(self.dim) / np.log(3))) == self.dim:
            self.radixes = [3] * int(np.round(np.log(self.dim) / np.log(3)))
        
        else:
            raise TypeError(
                "Unable to determine radixes"
                " for UnitaryMatrix with dim %d." % self.dim
            )
        
        if not is_valid_radixes(self.radixes):
            raise TypeError("Invalid qudit radixes.")

        self.size = len(self.radixes)

    @property
    def numpy(self) -> np.ndarray:
        return self.utry

    @property
    def shape(self) -> tuple[int, int]:
        return self.utry.shape

    def get_unitary(self, params: Sequence[float] = []) -> UnitaryMatrix:
        return self

    def get_dagger(self) -> UnitaryMatrix:
        """Returns the conjugate transpose of the unitary matrix."""
        return UnitaryMatrix(self.utry.conj().T)

    @staticmethod
    def identity(dim: int, radixes: Sequence[int] = []) -> UnitaryMatrix:
        """Returns an identity UnitaryMatrix."""
        if dim <= 0:
            raise ValueError('Invalid dimension for identity matrix.')
        return UnitaryMatrix(np.identity(dim), radixes)

    @staticmethod
    def closest_to(M: np.ndarray, radixes: Sequence[int] = []) -> UnitaryMatrix:
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
        return UnitaryMatrix(V @ Wh, radixes)
    
    def __matmul__(self, rhs: object) -> UnitaryMatrix:
        return UnitaryMatrix(self.numpy @ rhs, self.get_radixes())

    def save(self, filename: str) -> None:
        """Saves the unitary to a file."""
        np.savetxt(filename, self.utry)

    @staticmethod
    def from_file(filename: str) -> UnitaryMatrix:
        """Loads a unitary from a file."""
        return UnitaryMatrix(np.loadtxt(filename, dtype=np.complex128))


UnitaryLike = Union[UnitaryMatrix, np.ndarray, Sequence[Sequence[Any]]]
