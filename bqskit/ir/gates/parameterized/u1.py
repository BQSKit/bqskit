"""This module implements the U1Gate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class U1Gate(
    QubitGate,
    DifferentiableUnitary,
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

    _num_qudits = 1
    _num_params = 1
    _qasm_name = 'u1'

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        exp = np.exp(1j * params[0])

        return UnitaryMatrix(
            [
                [1, 0],
                [0, exp],
            ],
        )

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)

        dexp = 1j * np.exp(1j * params[0])

        return np.array(
            [
                [
                    [0, 0],
                    [0, dexp],
                ],
            ], dtype=np.complex128,
        )

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
