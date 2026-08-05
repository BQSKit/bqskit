"""This module implements the RYGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from openqudit.expressions import RYGate as _RYGate

from bqskit.ir.gate import Gate
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.utils.cachedclass import CachedClass


class RYGate(
    Gate,
    LocallyOptimizableUnitary,
    CachedClass,
):
    """
    A gate representing an arbitrary rotation around the Y axis.

    It is given by the following parameterized unitary:

    .. math::

        \\begin{pmatrix}
        \\cos{\\frac{\\theta}{2}} & -\\sin{\\frac{\\theta}{2}} \\\\
        \\sin{\\frac{\\theta}{2}} & \\cos{\\frac{\\theta}{2}} \\\\
        \\end{pmatrix}
    """

    _qasm_name = 'ry'
    _expr = _RYGate()

    def optimize(self, env_matrix: npt.NDArray[np.complex128]) -> list[float]:
        """
        Return the optimal parameters with respect to an environment matrix.

        See :class:`LocallyOptimizableUnitary` for more info.
        """
        self.check_env_matrix(env_matrix)
        a = np.real(env_matrix[0, 0] + env_matrix[1, 1])
        b = np.real(env_matrix[1, 0] - env_matrix[0, 1])
        theta = 2 * np.arccos(a / np.sqrt(a ** 2 + b ** 2))
        theta *= -1 if b > 0 else 1
        return [theta]
