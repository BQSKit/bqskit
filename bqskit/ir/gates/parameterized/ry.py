"""This module implements the RYGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.quditgate import QuditGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass
from bqskit.utils.typing import is_integer


class RYGate(
    QuditGate,
    DifferentiableUnitary,
    CachedClass,
):
    """
    A gate representing an arbitrary rotation around the Y axis. This is
    equivalent to rotation by the Y Pauli Gate in the subspace of 2 levels. 
    """

    _num_qudits = 1
    _num_params = 1
    _qasm_name = 'ry'

    def __init__(
        self, 
        num_levels: int = 2, 
        level_1: int = 0, 
        level_2: int = 1
    ) -> None:
        """
            Args:
            num_levels (int): The number of qudit levels (>=2).

            level_1 (int): the first level for the X qudit gate (<num_levels)
            level_2 (int): the second level for the X qudit gate (<num_levels)
            
            Raises:
            ValueError: if num_levels < 2
            ValueError: if any of levels >= num_levels
        """
        if num_levels < 2 or not is_integer(num_levels):
            raise ValueError(
                'RYGate num_levels must be a postive integer greater than or equal to 2.',
            )
        self.num_levels = num_levels
        if level_1 > num_levels or level_2 > num_levels:
            raise ValueError(
                'RYGate indices must be equal or less to the number of levels.',
            )
        self.level_1 = level_1
        self.level_2 = level_2

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""

        self.check_parameters(params)

        cos = np.cos(params[0] / 2)
        sin = np.sin(params[0] / 2)

        matrix = np.eye(self.num_levels, dtype=np.complex128)
        matrix[self.level_1, self.level_1] = cos
        matrix[self.level_2, self.level_2] = cos
        matrix[self.level_1, self.level_2] = -sin
        matrix[self.level_2, self.level_1] = sin

        return UnitaryMatrix(matrix, self.radixes)

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)

        dcos = -np.sin(params[0] / 2) / 2
        dsin = np.cos(params[0] / 2) / 2

        matrix = np.zeros(
            (self.num_levels, self.num_levels),
            dtype=np.complex128,
        )
        matrix[self.level_1, self.level_1] = dcos
        matrix[self.level_2, self.level_2] = dcos
        matrix[self.level_1, self.level_2] = -dsin
        matrix[self.level_2, self.level_1] = dsin

        return np.array([matrix], dtype=np.complex128)

#     def optimize(self, env_matrix: npt.NDArray[np.complex128]) -> list[float]:
#         """
#         Return the optimal parameters with respect to an environment matrix.

#         See :class:`LocallyOptimizableUnitary` for more info.
#         """
#         self.check_env_matrix(env_matrix)
#         a = np.real(env_matrix[0, 0] + env_matrix[1, 1])
#         b = np.real(env_matrix[1, 0] - env_matrix[0, 1])
#         theta = 2 * np.arccos(a / np.sqrt(a ** 2 + b ** 2))
#         theta *= -1 if b > 0 else 1
#         return [theta]
