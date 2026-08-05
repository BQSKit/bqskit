"""This module implements the ConstantGate base class."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gate import Gate
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.utils.cachedclass import CachedClass


class ConstantGate(
    Gate,
    LocallyOptimizableUnitary,
    CachedClass,
):
    """A gate that does not change during circuit instantiation."""

    _num_params = 0

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`Gate` for more info.
        """
        self.check_parameters(params)
        return np.array([])

    def optimize(self, env_matrix: npt.NDArray[np.complex128]) -> list[float]:
        """
        Return the optimal parameters with respect to an environment matrix.

        See :class:`LocallyOptimizableUnitary` for more info.
        """
        return []
