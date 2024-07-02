"""This module implements the PauliMatrices class."""
from __future__ import annotations

import itertools as it
from typing import Iterable
from typing import Iterator
from typing import overload
from typing import Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_numeric
from bqskit.utils.typing import is_sequence

if TYPE_CHECKING:
    from bqskit.qis.unitary.unitary import RealVector


class PauliZMatrices(Sequence[npt.NDArray[np.complex128]]):
    """
    The group of Pauli Z matrices.

    A PauliZMatrices object represents the entire of set of Pauli Z matrices for
    some number of qubits.
    """

    Z = np.array(
        [
            [1, 0],
            [0, -1],
        ], dtype=np.complex128,
    )
    """The Pauli Z Matrix."""

    I = np.array(
        [
            [1, 0],
            [0, 1],
        ], dtype=np.complex128,
    )
    """The Identity Matrix."""

    def __init__(self, num_qudits: int) -> None:
        """
        Construct the Pauli Z group for `num_qudits` number of qubits.

        Args:
            num_qudits (int): Power of the tensor product of the Pauli Z
                group.

        Raises:
            ValueError: If `num_qudits` is less than or equal to 0.
        """

        if not is_integer(num_qudits):
            raise TypeError(
                'Expected integer for num_qudits, got %s.' %
                type(num_qudits),
            )

        if num_qudits <= 0:
            raise ValueError(
                'Expected positive integer for num_qudits, got %s.' % type(
                    num_qudits,
                ),
            )

        self.num_qudits = num_qudits

        if num_qudits == 1:
            self.paulizs = [
                PauliZMatrices.I,
                PauliZMatrices.Z,
            ]
        else:
            self.paulizs = []
            matrices = it.product(
                PauliZMatrices(
                    num_qudits - 1,
                ),
                PauliZMatrices(1),
            )
            for pauliz_n_1, pauliz_1 in matrices:
                self.paulizs.append(np.kron(pauliz_n_1, pauliz_1))

    def __iter__(self) -> Iterator[npt.NDArray[np.complex128]]:
        return self.paulizs.__iter__()

    @overload
    def __getitem__(self, index: int) -> npt.NDArray[np.complex128]:
        ...

    @overload
    def __getitem__(self, index: slice) -> list[npt.NDArray[np.complex128]]:
        ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> npt.NDArray[np.complex128] | list[npt.NDArray[np.complex128]]:
        return self.paulizs[index]

    def __len__(self) -> int:
        return len(self.paulizs)

    @property
    def numpy(self) -> npt.NDArray[np.complex128]:
        """The NumPy array holding the pauliz matrices."""
        return np.array(self.paulizs)

    def __array__(
        self,
        dtype: np.typing.DTypeLike = np.complex128,
    ) -> npt.NDArray[np.complex128]:
        """Implements NumPy API for the PauliZMatrices class."""
        if dtype != np.complex128:
            raise ValueError('PauliZMatrices only supports Complex128 dtype.')

        return np.array(self.paulizs, dtype)

    def get_projection_matrices(
            self, q_set: Iterable[int],
    ) -> list[npt.NDArray[np.complex128]]:
        """
        Return the Pauli Z matrices that act only on qubits in `q_set`.

        Args:
            q_set (Iterable[int]): Active qubit indices

        Returns:
            list[np.ndarray]: Pauli Z matrices from `self` acting only
            on qubits in `q_set`.

        Raises:
            ValueError: if `q_set` is an invalid set of qubit indices.
        """
        q_set = list(q_set)

        if not all(is_integer(q) for q in q_set):
            raise TypeError('Expected sequence of integers for qubit indices.')

        if any(q < 0 or q >= self.num_qudits for q in q_set):
            raise ValueError('Qubit indices must be in [0, n).')

        if len(q_set) != len(set(q_set)):
            raise ValueError('Qubit indices cannot have duplicates.')

        # Nth Order Pauli Z Matrices can be thought of base 2 number
        # I = 0, Z = 1
        # IZZ = 1 * 2^2 + 1 * 2^1 + 0 * 4^0 = 6 (base 10)
        # This gives the idx of IZZ in paulizs
        # Note we read qubit index from the left,
        # so Z in ZII corresponds to q = 0
        pauliz_n_qubit = []
        for ps in it.product([0, 1], repeat=len(q_set)):
            idx = 0
            for p, q in zip(ps, q_set):
                idx += p * (2 ** (self.num_qudits - q - 1))
            pauliz_n_qubit.append(self.paulizs[idx])

        return pauliz_n_qubit

    def dot_product(self, alpha: RealVector) -> npt.NDArray[np.complex128]:
        """
        Computes the standard dot product of `alpha` with the paulis.

        Args:
            alpha (RealVector): The pauli coefficients.

        Returns:
            np.ndarray: Sum of element-wise multiplication of `alpha`
            and `self.paulis`.

        Raises:
            ValueError: If `alpha` and `self.paulis` are incompatible.
        """

        if not is_sequence(alpha) or not all(is_numeric(a) for a in alpha):
            msg = f'Expected a sequence of numbers, got {type(alpha)}.'
            raise TypeError(msg)

        if len(alpha) != len(self):
            msg = (
                'Incorrect number of alpha values, expected '
                f'{len(self)}, got {len(alpha)}.'
            )
            raise ValueError(msg)

        return np.array(np.sum([a * s for a, s in zip(alpha, self.paulizs)], 0))

    @staticmethod
    def from_string(
        pauliz_string: str,
    ) -> npt.NDArray[np.complex128] | list[npt.NDArray[np.complex128]]:
        """
        Construct Pauli Z matrices from a string description.

        Args:
            pauli_string (str): A string that describes the desired matrices.
                This is a comma-seperated list of pauli strings.
                A pauli string has the following regex pattern: [IZ]+

        Returns:
            np.ndarray | list[np.ndarray]: Either the single pauli Z matrix
            if only one is constructed, or the list of the constructed
            pauli Z matrices.

        Raises:
            ValueError: if `pauliz_string` is invalid.
        """

        if not isinstance(pauliz_string, str):
            msg = f'Expected str for pauliz_string, got {type(pauliz_string)}.'
            raise TypeError(msg)

        pauliz_strings = [
            string.strip().upper()
            for string in pauliz_string.split(',')
            if len(string.strip()) > 0
        ]

        pauliz_matrices = []
        idx_dict = {'I': 0, 'Z': 1}
        mat_dict = {
            'I': PauliZMatrices.I,
            'Z': PauliZMatrices.Z,
        }

        for pauli_string in pauliz_strings:
            if not all(char in 'IZ' for char in pauli_string):
                raise ValueError('Invalid Pauli Z string.')

            if len(pauli_string) <= 6:
                idx = 0
                for char in pauli_string:
                    idx *= 2
                    idx += idx_dict[char]
                pauliz_matrices.append(PauliZMatrices(len(pauli_string))[idx])
            else:
                acm = mat_dict[pauli_string[0]]
                for char in pauli_string[1:]:
                    acm = np.kron(acm, mat_dict[char])
                pauliz_matrices.append(acm)

        if len(pauliz_matrices) == 1:
            return pauliz_matrices[0]

        return pauliz_matrices
