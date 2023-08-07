"""This module implements the RZGate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates.quditgate import QuditGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass
from bqskit.utils.typing import is_integer


class RZGate(
    QuditGate, 
    DifferentiableUnitary, 
    CachedClass):
    """
    A gate representing an arbitrary rotation by the Z qudit gate This is
    equivalent to rotation by the Z Pauli Gate in the subspace of 2 levels.
    """

    _num_qudits = 1
    _num_params = 1
    _qasm_name = 'rz'

    def __init__(
        self, 
        num_levels: int = 2, 
        level_1: int=0,
        level_2: int = 1):
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
                'RXGate num_levels must be a postive integer greater than or equal to 2.',
            )
        self.num_levels = num_levels
        if level_1 > num_levels or level_2 > num_levels:
            raise ValueError(
                'RXGate indices must be equal or less to the number of levels.',
            )
        self.level_1 = level_1
        self.level_2 = level_2

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        matrix = np.eye(self.num_levels, dtype=complex) * \
            np.exp(-params[0] / 2 * 1j)
        matrix[level_2, level_2] = np.exp(params[0] / 2 * 1j)
        return UnitaryMatrix(matrix, self.radixes)

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)

        matrix = np.eye(self.num_levels, dtype=complex) * \
            np.exp(-params[0] / 2 * 1j) * (-0.5 * 1j)
        matrix[level_2, level_2] = np.exp(params[0] / 2 * 1j) * (0.5 * 1j)

        return np.array(
            [
                matrix,
            ], dtype=np.complex128,
        )
