"""This module implements the GeneralGate base class."""
from __future__ import annotations
import abc
from typing import Sequence

import numpy as np
import numpy.typing as npt
import scipy as sp

from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class GeneralGate(LocallyOptimizableUnitary):
    """An abstract base class for gates that parameterize any unitary."""

    @abc.abstractmethod
    def calc_params(self, utry: UnitaryMatrix) -> list[float]:
        """Return the parameters for this gate to implement `utry`."""

    def identity_as_params(self, radixes: Sequence[int]) -> list[float]:
        """Return the parameters for the gate that implements the identity."""
        identity = UnitaryMatrix.identity(np.prod(radixes), radixes)
        return self.calc_params(identity)

    def optimize(self, env_matrix: npt.NDArray[np.complex128]) -> list[float]:
        """
        Return the optimal parameters with respect to an environment matrix.

        See :class:`LocallyOptimizableUnitary` for more info.
        """
        self.check_env_matrix(env_matrix)
        U, _, Vh = sp.linalg.svd(env_matrix)
        new_U = Vh.conj().T @ U.conj().T
        return self.calc_params(UnitaryMatrix(new_U))
