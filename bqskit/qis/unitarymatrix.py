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
from bqskit.utils.typing import is_square_matrix
from bqskit.utils.typing import is_unitary


class UnitaryMatrix(Unitary):
    """The UnitaryMatrix Class."""

    def __init__(self, utry: UnitaryLike) -> None:
        """Constructs a UnitaryMatrix with the supplied unitary matrix."""
        np_utry = np.array(utry, dtype=np.complex128)

        if not is_unitary(np_utry):
            raise TypeError('Expected unitary matrix.')

        self.utry = np_utry

    @property
    def numpy(self) -> np.ndarray:
        return self.utry

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.utry.shape

    def get_unitary(self, params: Sequence[float] = []) -> UnitaryMatrix:
        return self

    def is_qubit_unitary(self) -> bool:
        """Returns true if this unitary can represent a qubit system."""
        return self.utry.shape[0] & (self.utry.shape[0] - 1) == 0

    def get_num_qubits(self) -> int:
        """Returns the number of qubits this unitary can represent."""
        if not self.is_qubit_unitary():
            raise TypeError('Unitary does not represent a pure qubit system.')
        return int(np.log2(len(self.utry)))

    def dagger(self) -> UnitaryMatrix:
        """Returns the conjugate transpose of the unitary matrix."""
        return UnitaryMatrix(self.utry.conj().T)

    @staticmethod
    def identity(dim: int) -> UnitaryMatrix:
        """Returns an identity UnitaryMatrix."""
        if dim <= 0:
            raise ValueError('Invalid dimension for identity matrix.')
        return UnitaryMatrix(np.identity(dim))

    @staticmethod
    def closest_to(M: np.ndarray) -> UnitaryMatrix:
        """
        Calculate and return the closest unitary to a given matrix.

        Calculate the unitary matrix U that is closest with respect to the
        operator norm distance to the general matrix M.

        D.M.Reich. “Characterisation and Identification of Unitary Dynamics
        Maps in Terms of Their Action on Density Matrices”

        Args:
            M (np.ndarray): The matrix input.
        Returns:
            (np.ndarray): The unitary matrix closest to M.
        """

        if not is_square_matrix(M):
            raise TypeError('Expected square matrix.')

        V, _, Wh = sp.linalg.svd(M)
        return V @ Wh

    def save(self, filename: str) -> None:
        """Saves the unitary to a file."""
        np.savetxt(filename, self.utry)

    @staticmethod
    def from_file(filename: str) -> UnitaryMatrix:
        """Loads a unitary from a file."""
        return UnitaryMatrix(np.loadtxt(filename, dtype=np.complex128))


UnitaryLike = Union[UnitaryMatrix, np.ndarray, Sequence[Sequence[Any]]]
