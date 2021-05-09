"""
This module implements the PauliMatrices class.

A PauliMatrices object represents the entire of set of pauli matrices for some
number of qubits.

"""
from __future__ import annotations

import itertools as it
from typing import Iterator
from typing import overload
from typing import Sequence

import numpy as np
import scipy as sp

from bqskit.utils.cachedclass import CachedClass
from bqskit.utils.typing import is_hermitian
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_numeric
from bqskit.utils.typing import is_sequence
from bqskit.utils.typing import is_unitary


class PauliMatrices(CachedClass, Sequence[np.ndarray]):
    """The PauliMatrices class."""

    # The Pauli Matrices
    X = np.array(
        [
            [0, 1],
            [1, 0],
        ], dtype=np.complex128,
    )

    Y = np.array(
        [
            [0, -1j],
            [1j, 0],
        ], dtype=np.complex128,
    )

    Z = np.array(
        [
            [1, 0],
            [0, -1],
        ], dtype=np.complex128,
    )

    I = np.array(
        [
            [1, 0],
            [0, 1],
        ], dtype=np.complex128,
    )

    def __init__(self, size: int) -> None:
        """
        Construct the Pauli group for `size` number of qubits.

        Args:
            size (int): Power of the tensor product of the Pauli group.

        Raises:
            ValueError: If `size` is less than or equal to 0.

        """

        if not is_integer(size):
            raise TypeError('Expected integer for size, got %s.' % type(size))

        if size <= 0:
            raise ValueError(
                'Expected positive integer for size, got %s.' % type(size),
            )

        self.size = size

        if size == 1:
            self.paulis = [
                PauliMatrices.I,
                PauliMatrices.X,
                PauliMatrices.Y,
                PauliMatrices.Z,
            ]
        else:
            self.paulis = []
            matrices = it.product(PauliMatrices(size - 1), PauliMatrices(1))
            for pauli_n_1, pauli_1 in matrices:
                self.paulis.append(np.kron(pauli_n_1, pauli_1))

    def __iter__(self) -> Iterator[np.ndarray]:
        return self.paulis.__iter__()

    @overload
    def __getitem__(self, index: int) -> np.ndarray:
        ...

    @overload
    def __getitem__(self, index: slice) -> list[np.ndarray]:
        ...

    def __getitem__(self, index: int | slice) -> np.ndarray | list[np.ndarray]:
        return self.paulis[index]

    def __len__(self) -> int:
        return len(self.paulis)

    def get_numpy(self) -> np.ndarray:
        return np.array(self.paulis)

    def get_projection_matrices(
            self, q_set: Sequence[int],
    ) -> list[np.ndarray]:
        """
        Return the Pauli matrices that act only on qubits in `q_set`.

        Args:
            q_set (Sequence[int]): Active qubit indices

        Returns:
            (list(np.ndarray)): Pauli matrices from `self` acting only
                on qubits in `q_set`.

        Raises:
            ValueError: if `q_set` is an invalid set of qubit indices.

        """

        if not all(is_integer(q) for q in q_set):
            raise TypeError('Expected sequence of integers for qubit indices.')

        if any(q < 0 or q >= self.size for q in q_set):
            raise ValueError('Qubit indices must be in [0, n).')

        if len(q_set) != len(set(q_set)):
            raise ValueError('Qubit indices cannot have duplicates.')

        if len(q_set) == 0:
            raise ValueError('Need at least one qubit index.')

        # Nth Order Pauli Matrices can be thought of base 4 number
        # I = 0, X = 1, Y = 2, Z = 3
        # XXY = 1 * 4^2 + 1 * 4^1 + 2 * 4^0 = 22 (base 10)
        # This gives the idx of XXY in paulis
        # Note we read qubit index from the left,
        # so X in XII corresponds to q = 0
        pauli_n_qubit = []
        for ps in it.product([0, 1, 2, 3], repeat=len(q_set)):
            idx = 0
            for p, q in zip(ps, q_set):
                idx += p * (4 ** (self.size - q - 1))
            pauli_n_qubit.append(self.paulis[idx])

        return pauli_n_qubit

    def dot_product(self, alpha: Sequence[float] | np.ndarray) -> np.ndarray:
        """
        Computes the standard dot product of `alpha` with the paulis.

        Args:
            alpha (Sequence[float] | np.ndarray): The pauli coefficients.

        Returns:
            (np.ndarray): Sum of element-wise multiplication of `alpha`
                and `self.paulis`.

        Raises:
            ValueError: If `alpha` and `self.paulis` are incompatible.

        """

        if not is_sequence(alpha) or not all(is_numeric(a) for a in alpha):
            raise TypeError(
                'Expected a sequence of numbers, got %s.' % type(alpha),
            )

        if len(alpha) != len(self):
            raise ValueError(
                'Incorrect number of alpha values, expected %d, got %d.'
                % (len(self), len(alpha)),
            )

        return np.array(np.sum([a * s for a, s in zip(alpha, self.paulis)], 0))

    @staticmethod
    def from_string(pauli_string: str) -> np.ndarray | list[np.ndarray]:
        """
        Construct pauli matrices from a string description.

        Args:
            pauli_string (str): A string that describes the desired matrices.
                This is a comma-seperated list of pauli strings.
                A pauli string has the following regex pattern:
                    [IXYZ]+

        Returns:
            (np.ndarray | list(np.ndarray)):
                Either the single pauli matrix if only one is constructed,
                or the list of the constructed pauli matrices.

        Raises:
            ValueError: if `pauli_string` is invalid.

        """

        if not isinstance(pauli_string, str):
            raise TypeError(
                'Expected string for pauli_string, got %s' % type(
                    pauli_string,
                ),
            )

        pauli_strings = [
            string.strip().upper()
            for string in pauli_string.split(',')
            if len(string.strip()) > 0
        ]

        pauli_matrices = []
        idx_dict = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
        mat_dict = {
            'I': PauliMatrices.I,
            'X': PauliMatrices.X,
            'Y': PauliMatrices.Y,
            'Z': PauliMatrices.Z,
        }

        for pauli_string in pauli_strings:
            if not all(char in 'IXYZ' for char in pauli_string):
                raise ValueError('Invalid pauli string.')

            if len(pauli_string) <= 6:
                idx = 0
                for char in pauli_string:
                    idx *= 4
                    idx += idx_dict[char]
                pauli_matrices.append(PauliMatrices(len(pauli_string))[idx])
            else:
                acm = mat_dict[pauli_string[0]]
                for char in pauli_string[1:]:
                    acm = np.kron(acm, mat_dict[char])
                pauli_matrices.append(acm)

        if len(pauli_matrices) == 1:
            return pauli_matrices[0]

        return pauli_matrices
