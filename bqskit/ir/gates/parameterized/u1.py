"""This module implements the U1Gate."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from openqudit.expressions import U1Gate as _U1Gate

from bqskit.ir.gate import Gate
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.utils.cachedclass import CachedClass


class U1Gate(
    Gate,
    LocallyOptimizableUnitary,
    CachedClass,
):
    """
    The U1 single qubit gate.

    It is given by the following parameterized unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 \\\\
        0 & \\exp({i\\theta}) \\\\
        \\end{pmatrix}
    """

    _qasm_name = 'u1'
    _expr = _U1Gate()

    def optimize(self, env_matrix: npt.NDArray[np.complex128]) -> list[float]:
        """
        Return the optimal parameters with respect to an environment matrix.

        See :class:`LocallyOptimizableUnitary` for more info.
        """
        self.check_env_matrix(env_matrix)
        a = np.real(env_matrix[1, 1])
        b = np.imag(env_matrix[1, 1])
        arctan = np.arctan(b / a)

        if a < 0 and b > 0:
            arctan += np.pi
        elif a < 0 and b < 0:
            arctan -= np.pi

        return [-arctan]
