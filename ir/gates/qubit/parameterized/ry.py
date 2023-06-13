"""This module implements the RYGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class RYGate(
    QubitGate,
    DifferentiableUnitary,
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

    _num_qudits = 1
    _num_params = 1
    _qasm_name = 'ry'

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        cos = np.cos(params[0] / 2)
        sin = np.sin(params[0] / 2)

        return UnitaryMatrix(
            [
                [cos, -sin],
                [sin, cos],
            ],
        )

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)

        dcos = -np.sin(params[0] / 2) / 2
        dsin = np.cos(params[0] / 2) / 2

        return np.array(
            [
                [
                    [dcos, -dsin],
                    [dsin, dcos],
                ],
            ], dtype=np.complex128,
        )

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
