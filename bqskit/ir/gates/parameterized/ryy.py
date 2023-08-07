"""This module implements the RYYGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.quditgate import QuditGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass
from bqskit.utils.typing import is_integer


class RYYGate(
    QuditGate,
    DifferentiableUnitary,
    CachedClass,
):
    """
    A qudit gate representing an arbitrary rotation around the YY axis for qudits.
    """

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'ryy'

    def __init__(
        self, 
        num_levels: int = 2, 
        level_1: int = 0, 
        level_2: int = 1, 
        level_3: int = 0, 
        level_4: int = 1
    ) ->None:
        """
            Args:
            num_levels (int): The number of qudit levels (>=2).

            level_1 (int): the first level for the first Y qudit gate (<num_levels)
            level_2 (int): the second level for the first Y qudit gate (<num_levels)
            level_3 (int): the first level for the second Y qudit gate (<num_levels)
            level_4 (int): the second level for the second Y qudit gate (<num_levels) 
            
            Raises:
            ValueError: if num_levels < 2
            ValueError: if any of levels >= num_levels
        """
        if num_levels < 2 or not is_integer(num_levels):
            raise ValueError(
                'RYYGate num_levels must be a postive integer greater than or equal to 2.',
            )
        self.num_levels = num_levels
        if level_1 > num_levels or level_2 > num_levels or level_3 > num_levels or level_4 > num_levels:
            raise ValueError(
                'RYYGate indices must be equal or less to the number of levels.',
            )
        self.level_1 = level_1
        self.level_2 = level_2
        self.level_3 = level_3
        self.level_4 = level_4

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        cos = np.cos(params[0] / 2)
        nsin = -1j * np.sin(params[0] / 2)
        psin = 1j * np.sin(params[0] / 2)

        idx1 = self.level_1 * self.num_levels + self.level_3
        idx2 = self.level_1 * self.num_levels + self.level_4 
        idx3 = self.level_2 * self.num_levels + self.level_3
        idx4 = self.level_2 * self.num_levels + self.level_4

        matrix = np.eye(self.num_levels, dtype=np.complex128)
        matrix[idx1,idx1] = cos
        matrix[idx2,idx2] = cos
        matrix[idx3,idx3] = cos
        matrix[idx4,idx4] = cos

        matrix[idx1,idx4] = psin
        matrix[idx1,idx4] = nsin
        matrix[idx4,idx1] = nsin
        matrix[idx4,idx1] = psin

        return UnitaryMatrix(matrix, self.radixes)

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)

        dcos = -np.sin(params[0] / 2) / 2
        dnsin = -1j * np.cos(params[0] / 2) / 2
        dpsin = 1j * np.cos(params[0] / 2) / 2

        idx1 = self.level_1 * self.num_levels + self.level_3
        idx2 = self.level_1 * self.num_levels + self.level_4 
        idx3 = self.level_2 * self.num_levels + self.level_3
        idx4 = self.level_2 * self.num_levels + self.level_4 

        matrix = np.zeros(
            (self.num_levels, self.num_levels),
            dtype=np.complex128,
        )
        matrix[idx1,idx1] = dcos
        matrix[idx2,idx2] = dcos
        matrix[idx3,idx3] = dcos
        matrix[idx4,idx4] = dcos

        matrix[idx1,idx4] = dpsin
        matrix[idx1,idx4] = dnsin
        matrix[idx4,idx1] = dnsin
        matrix[idx4,idx1] = dpsin

        return np.array([matrix], dtype=np.complex128)
