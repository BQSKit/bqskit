"""This module implements the rotations due to the generatos of SU(3)."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.qutritgate import QutritGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass
from bqskit.utils.typing import is_integer


class RSU3Gate(QutritGate, DifferentiableUnitary, CachedClass):
    """Rotation by SU3 generator for a single qutrit gate.
    
        .. math::
        \\exp(i * params[0] * \lambda_j)

        where lambda_j is the j-th generator of the SU(3) Lie algebra.
        We use the physics notation, so each generator is Hermitian (not unitary).
        There are N^2-1 = 3^2-1 =8 generators, and N-1-=3-1=2 generators that coomutate.

        Reference: "Lie algebras in particle physics : from isospin to unified theories", Howard Georgi 
    """

    _num_qudits = 1
    _num_params = 1
    num_levels = 3

    def __init__(self, index: int):
        """
            Raises:
            TypeError: If index is not an integer
            
            ValueError: If index is less than 0 or greater than 7
        """
        if not is_integer(index):
           raise TypeError(
                'RSU3Gate generator index must be an integer.',
            )
        if index < 0 or index > 7:
            raise ValueError(
                'RSU3Gate generator index must be a non negative integer between 0 and 7.',
            )
        self.index = index

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""

        self.check_parameters(params)

        cos = np.cos(params[0])
        sin = np.sin(params[0])

        matrix = np.eye(self.num_levels, dtype=np.complex128)

        if self.index == 0:
            matrix[0, 0] = cos
            matrix[1, 1] = cos
            matrix[0, 1] = -1j * sin
            matrix[1, 0] = -1j * sin
        elif self.index == 1:
            matrix[0, 0] = cos
            matrix[1, 1] = cos
            matrix[0, 1] = -sin
            matrix[1, 0] = sin
        elif self.index == 2:
            matrix[0, 0] = np.exp(-1j * params[0])
            matrix[1, 1] = np.exp(1j * params[0])
        elif self.index == 3:
            matrix[0, 0] = cos
            matrix[2, 2] = cos
            matrix[0, 2] = -1j * sin
            matrix[2, 0] = -1j * sin
        elif self.index == 4:
            matrix[0, 0] = cos
            matrix[2, 2] = cos
            matrix[0, 2] = -sin
            matrix[2, 0] = sin
        elif self.index == 5:
            matrix[1, 1] = cos
            matrix[2, 2] = cos
            matrix[1, 2] = -1j * sin
            matrix[2, 1] = -1j * sin
        elif self.index == 6:
            matrix[1, 1] = cos
            matrix[2, 2] = cos
            matrix[1, 2] = -sin
            matrix[2, 1] = sin
        elif self.index == 7:
            matrix[0, 0] = np.exp(-1j * params[0] / np.sqrt(3))
            matrix[1, 1] = np.exp(-1j * params[0] / np.sqrt(3))
            matrix[2, 2] = np.exp(2j * params[0] / np.sqrt(3))
        return UnitaryMatrix(matrix, self.radixes)

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)

        cos = np.cos(params[0])
        sin = np.sin(params[0])

        matrix = np.zeros(
            (self.num_levels, self.num_levels),
            dtype=np.complex128,
        )
        if self.index == 0:
            matrix[0, 0] = -sin
            matrix[1, 1] = -sin
            matrix[0, 1] = -1j * cos
            matrix[1, 0] = -1j * cos
        elif self.index == 1:
            matrix[0, 0] = -sin
            matrix[1, 1] = -sin
            matrix[0, 1] = -cos
            matrix[1, 0] = cos
        elif self.index == 2:
            matrix[0, 0] = -1j * np.exp(-1j * params[0])
            matrix[1, 1] = 1j * np.exp(1j * params[0])
        elif self.index == 3:
            matrix[0, 0] = -sin
            matrix[2, 2] = -sin
            matrix[0, 2] = -1j * cos
            matrix[2, 0] = -1j * cos
        elif self.index == 4:
            matrix[0, 0] = -sin
            matrix[2, 2] = -sin
            matrix[0, 2] = -cos
            matrix[2, 0] = cos
        elif self.index == 5:
            matrix[1, 1] = sin
            matrix[2, 2] = sin
            matrix[1, 2] = -1j * cos
            matrix[2, 1] = -1j * cos
        elif self.index == 6:
            matrix[1, 1] = sin
            matrix[2, 2] = sin
            matrix[1, 2] = -cos
            matrix[2, 1] = cos
        elif self.index == 7:
            matrix[0, 0] = -1j / np.sqrt(3) * \
                np.exp(-1j * params[0] / np.sqrt(3))
            matrix[1, 1] = -1j / np.sqrt(3) * \
                np.exp(-1j * params[0] / np.sqrt(3))
            matrix[2, 2] = 2j / np.sqrt(3) * np.exp(2j * params[0] / np.sqrt(3))
        return np.array([matrix], dtype=np.complex128)