"""This module implements the LocallyOptimizableUnitary base class."""
from __future__ import annotations

import abc

import numpy as np

from bqskit.qis.unitary.unitary import Unitary
from bqskit.utils.typing import is_square_matrix


class LocallyOptimizableUnitary(Unitary):
    """
    The LocallyOptimizableUnitary base class.

    A LocallyOptimizableUnitary exposes the `optimize` abstract method.
    """

    @abc.abstractmethod
    def optimize(self, env_matrix: np.ndarray) -> list[float]:
        """
        Return the optimal parameters with respect to an environment matrix.

        More specifically, return the parameters that maximize the
        real component of the trace of the product between `env_matrix`
        and this unitary:
            `argmax(Re(Trace(env_matrix @ self.get_unitary(params))))`

        Args:
            env_matrix (np.ndarray): Optimize with respect to this matrix.
                Has the same dimensions as this unitary.

        Returns
            (list[float]): The parameters that optimizes this unitary.
        """

    def check_env_matrix(self, env_matrix: np.ndarray) -> None:
        """Check to ensure the `env_matrix` is valid and matches the gate."""
        if not is_square_matrix(env_matrix):
            raise TypeError('Expected a square matrix.')

        if env_matrix.shape != (self.get_dim(), self.get_dim()):
            raise TypeError('Environmental matrix shape mismatch.')
