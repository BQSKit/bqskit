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


class Unitary(metaclass=UnitaryMeta):
    """
    A unitary-valued function.

    A `Unitary` is map from zero or more real numbers to a unitary matrix. This
    is captured in the main `get_unitary` abstract method.
    """

    _num_params: int
    _num_qudits: int
    _radixes: tuple[int, ...]
    _dim: int

    @property
    def num_params(self) -> int:
        """The number of parameters for this unitary."""
        return getattr(self, '_num_params')

    @property
    def num_qudits(self) -> int:
        """The number of qudits this unitary can act on."""
        return getattr(self, '_num_qudits', len(self.radixes))

    @property
    def radixes(self) -> tuple[int, ...]:
        """The number of orthogonal states for each qudit."""
        return getattr(self, '_radixes')

    @property
    def dim(self) -> int:
        """The matrix dimension for this unitary."""
        return getattr(self, '_dim', int(np.prod(self.radixes)))

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
        return all([radix == 2 for radix in self.radixes])

    def is_qutrit_only(self) -> bool:
        """Returns true if this unitary can only act on qutrits."""
        return all([radix == 3 for radix in self.radixes])

    def is_qudit_only(self, radix: int) -> bool:
        """Returns true if this unitary can only act on `radix`-qudits."""
        return all([r == radix for r in self.radixes])

    def is_parameterized(self) -> bool:
        """Returns true if this unitary is parameterized."""
        return self.num_params != 0

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

        if len(params) != self.num_params:
            raise ValueError(
                'Expected %d params, got %d.'
                % (self.num_params, len(params)),
            )
