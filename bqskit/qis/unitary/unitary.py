"""This module implements the Unitary abstract base class."""
from __future__ import annotations

import abc
from typing import Sequence
from typing import TYPE_CHECKING

import numpy as np

from bqskit.qis.unitary.meta import UnitaryMeta
from bqskit.utils.typing import is_numeric
from bqskit.utils.typing import is_sequence

if TYPE_CHECKING:
    from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class Unitary (metaclass=UnitaryMeta):
    """
    The Unitary base class.

    A Unitary is map from zero or more real numbers to a unitary matrix. This is
    captured in the main `get_unitary` abstract method.
    """

    num_params: int
    radixes: tuple[int, ...]
    size: int
    dim: int

    def get_num_params(self) -> int:
        """Returns the number of parameters for this unitary."""
        if hasattr(self, 'num_params'):
            return self.num_params

        raise AttributeError(
            'Expected num_params field for unitary'
            ': %s.' % self.__class__.__name__,
        )

    def get_radixes(self) -> tuple[int, ...]:
        """Returns the number of orthogonal states for each qudit."""
        if hasattr(self, 'radixes'):
            return self.radixes

        raise AttributeError(
            'Expected radixes field for unitary'
            ': %s.' % self.__class__.__name__,
        )

    def get_size(self) -> int:
        """Returns the number of qudits this unitary can act on."""
        if hasattr(self, 'size'):
            return self.size

        raise AttributeError(
            'Expected size field for unitary'
            ': %s.' % self.__class__.__name__,
        )

    def get_dim(self) -> int:
        """Returns the matrix dimension for this unitary."""
        if hasattr(self, 'dim'):
            return self.dim

        return int(np.prod(self.get_radixes()))

    @abc.abstractmethod
    def get_unitary(self, params: Sequence[float] = []) -> UnitaryMatrix:
        """
        Abstract method that should return this unitary as a UnitaryMatrix.

        Args:
            params (Sequence[float]): Unconstrained real number
                parameters for parameterized unitaries.

        Returns:
            (UnitaryMatrix): The unitary matrix.
        """

    def is_qubit_only(self) -> bool:
        """Returns true if this unitary can only act on qubits."""
        return all([radix == 2 for radix in self.get_radixes()])

    def is_qutrit_only(self) -> bool:
        """Returns true if this unitary can only act on qutrits."""
        return all([radix == 3 for radix in self.get_radixes()])

    def is_parameterized(self) -> bool:
        """Returns true if this unitary is parameterized."""
        return self.get_num_params() != 0

    def is_constant(self) -> bool:
        """Returns true if this unitary doesn't have parameters."""
        return not self.is_parameterized()

    def check_parameters(self, params: Sequence[float] | np.ndarray) -> None:
        """Checks to ensure parameters are valid and match the unitary."""
        if not is_sequence(params):
            raise TypeError(
                'Expected a sequence type for params, got %s.'
                % type(params),
            )

        if not all(is_numeric(p) for p in params):
            typechecks = [is_numeric(p) for p in params]
            fail_idx = typechecks.index(False)
            raise TypeError(
                'Expected params to be floats, got %s.'
                % type(params[fail_idx]),
            )

        if len(params) != self.get_num_params():
            raise ValueError(
                'Expected %d params, got %d.'
                % (self.get_num_params(), len(params)),
            )
