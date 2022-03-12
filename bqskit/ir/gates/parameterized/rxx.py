"""This module implements the RXXGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class RXXGate(
    QubitGate,
    DifferentiableUnitary,
    LocallyOptimizableUnitary,
    CachedClass,
):
    """
    A gate representing an arbitrary rotation around the XX axis.

    It is given by the following parameterized unitary:

    .. math::

        \\begin{pmatrix}
        \\cos{\\frac{\\theta}{2}} & 0 & 0 & -\\sin{\\frac{\\theta}{2}}i \\\\
        0 & \\cos{\\frac{\\theta}{2}} & -\\sin{\\frac{\\theta}{2}}i & 0 \\\\
        0 & -\\sin{\\frac{\\theta}{2}}i & \\cos{\\frac{\\theta}{2}} & 0 \\\\
        -\\sin{\\frac{\\theta}{2}}i & 0 & 0 & \\cos{\\frac{\\theta}{2}} \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'rxx'

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        cos = np.cos(params[0] / 2)
        sin = -1j * np.sin(params[0] / 2)

        return UnitaryMatrix(
            [
                [cos, 0, 0, sin],
                [0, cos, sin, 0],
                [0, sin, cos, 0],
                [sin, 0, 0, cos],
            ],
        )

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)

        dcos = -np.sin(params[0] / 2) / 2
        dsin = -1j * np.cos(params[0] / 2) / 2

        return np.array(
            [
                [
                    [dcos, 0, 0, dsin],
                    [0, dcos, dsin, 0],
                    [0, dsin, dcos, 0],
                    [dsin, 0, 0, dcos],
                ],
            ], dtype=np.complex128,
        )

    def optimize(self, env_matrix: npt.NDArray[np.complex128]) -> list[float]:
        """
        Return the optimal parameters with respect to an environment matrix.

        See :class:`LocallyOptimizableUnitary` for more info.
        """
        self.check_env_matrix(env_matrix)
        a = np.real(
            env_matrix[0, 0] + env_matrix[1, 1]
            + env_matrix[2, 2] + env_matrix[3, 3],
        )
        b = np.imag(
            env_matrix[0, 3] + env_matrix[1, 2]
            + env_matrix[2, 1] + env_matrix[3, 0],
        )
        theta = np.arccos(a / np.sqrt(a ** 2 + b ** 2))
        theta *= -2 if b < 0 else 2
        return [theta]
