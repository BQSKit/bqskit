"""This module implements the LocallyOptimizableUnitary base class."""
from __future__ import annotations

import abc

import numpy as np
import numpy.typing as npt

from bqskit.qis.unitary.unitary import Unitary
from bqskit.utils.typing import is_square_matrix


class LocallyOptimizableUnitary(Unitary):
    """
    A locally optimizable unitary-valued function.

    A locally optimizable unitary-valued function is one that can be optimized
    with respect to a fixed environment. A `LocallyOptimizableUnitary` inherits
    from `Unitary` and additionally exposes the :func:`optimize` abstract
    method.
    """

    @abc.abstractmethod
    def optimize(self, env_matrix: npt.NDArray[np.complex128]) -> list[float]:
        """
        Return the optimal parameters with respect to an environment matrix.

        More specifically, return the parameters that maximize the
        real component of the trace of the product between `env_matrix`
        and this unitary:

        .. math::

            argmax(Re(Tr(\\mathit{env\\_matrix} \\times
            \\mathit{self.get\\_unitary(params))}))

        Args:
            env_matrix (npt.NDArray[np.complex128]): Optimize with respect
                to this matrix. Has the same dimensions as this unitary.

        Returns:
            list[float]: The parameters that optimizes this unitary.
        """

    def check_env_matrix(self, env_matrix: npt.NDArray[np.complex128]) -> None:
        """Check to ensure the `env_matrix` is validly shaped."""
        if not is_square_matrix(env_matrix):
            raise TypeError('Expected a square matrix.')

        if env_matrix.shape != (self.dim, self.dim):
            raise TypeError('Environmental matrix shape mismatch.')
