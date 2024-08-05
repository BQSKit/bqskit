"""This module implements the Unitary abstract base class."""
from __future__ import annotations

import abc
from typing import Sequence
from typing import TYPE_CHECKING
from typing import Union

import numpy as np
import numpy.typing as npt

from bqskit.qis.unitary.meta import UnitaryMeta
from bqskit.utils.typing import is_real_number
from bqskit.utils.typing import is_sequence

if TYPE_CHECKING:
    from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class Unitary(metaclass=UnitaryMeta):
    """
    A unitary-valued function.

    A `Unitary` is a map from zero or more real numbers to a unitary matrix.
    This is captured in the `get_unitary` abstract method.
    """

    _num_params: int
    _num_qudits: int
    _radixes: tuple[int, ...]
    _dim: int

    @property
    def num_params(self) -> int:
        """The number of real parameters this unitary-valued function takes."""
        return getattr(self, '_num_params')

    @property
    def num_qudits(self) -> int:
        """The number of qudits this unitary can act on."""
        if hasattr(self, '_num_qudits'):
            return self._num_qudits

        return len(self.radixes)

    @property
    def radixes(self) -> tuple[int, ...]:
        """The number of orthogonal states for each qudit."""
        return getattr(self, '_radixes')

    @property
    def dim(self) -> int:
        """The matrix dimension for this unitary."""
        if hasattr(self, '_dim'):
            return self._dim

        # return int(np.prod(self.radixes))
        # Above line removed due to failure to handle overflow and
        # underflows for large dimensions.

        acm = 1
        for radix in self.radixes:
            acm *= radix
        return acm


    @abc.abstractmethod
    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """
        Map real-valued `params` to a `UnitaryMatrix`.

        Args:
            params (RealVector): Unconstrained vector of real number
                parameters for parameterized unitaries.

        Returns:
            UnitaryMatrix: The unitary matrix.
        """

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

    def is_parameterized(self) -> bool:
        """Return true if this unitary is parameterized."""
        return self.num_params != 0

    def is_constant(self) -> bool:
        """Return true if this unitary doesn't take parameters."""
        return not self.is_parameterized()

    def check_parameters(self, params: RealVector) -> None:
        """
        Check parameters are valid and match the unitary.

        Args:
            params(RealVector): The parameters to check.

        Raises:
            ValueError: If parameter length does not match expected number.
        """
        if not is_sequence(params):
            raise TypeError(
                'Expected a sequence type for params, got %s.'
                % type(params),
            )

        if not all(is_real_number(p) for p in params):
            typechecks = [is_real_number(p) for p in params]
            fail_idx = typechecks.index(False)
            raise TypeError(
                'Expected params to be floats, got %s.'
                % type(params[fail_idx]),
            )

        if len(params) != self.num_params:
            raise ValueError(
                'Expected %d params, got %d.'
                % (self.num_params, len(params)),
            )

    def is_self_inverse(self, params: RealVector = []) -> bool:
        """
        Checks whether the unitary is its own inverse.

        A unitary is its own inverse if its matrix is equal to its
        Hermitian conjugate.

        Args:
            params (RealVector): The parameters of the unitary to check.

        Returns:
            bool: True if the unitary is self-inverse, False otherwise.

        Note:
            - This checks that the unitary is self-inverse for the given
              parameters only.
        """
        # Get the unitary matrix of the gate
        unitary_matrix = self.get_unitary(params)

        # Calculate the Hermitian conjugate (adjoint) of the unitary matrix
        hermitian_conjugate = unitary_matrix.dagger

        # Check if the unitary matrix is equal to its Hermitian conjugate
        return np.allclose(unitary_matrix, hermitian_conjugate)


RealVector = Union[
    Sequence[float],
    npt.NDArray[np.float64],
    npt.NDArray[np.float32],
]
