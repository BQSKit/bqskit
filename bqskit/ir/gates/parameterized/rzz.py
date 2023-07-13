"""This module implements the RZZGate.""" 
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.quditgate import QuditGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass
from bqskit.utils.typing import is_integer

class RZZGate(
    QuditGate,
    DifferentiableUnitary,
    CachedClass,
):
    """
    A gate representing an arbitrary rotation around the ZZ axis for qudits.
    This is equivalent to rotation by the Z Pauli Gate in the subspace of 2 levels
    
    __init__() arguments:
        num_levels : int
            Number of levels in each qudit (d).
        level: int
             The level on which to apply the Z gate (0...d-1).
    get_unitary arguments:
            param: float
            The angle by which to rotate
    
    """

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'rzz'
    
    def __init__(self, num_levels: int=2, level_1: int=1, level_2: int=1):
        if num_levels <2 or not is_integer(num_levels):
            raise ValueError(
                'RZGate num_levels must be a postive integer greater than or equal to 2.',
            )
        self.num_levels = num_levels
        if level_1 > num_levels or level_2 > num_levels:
            raise ValueError(
                'ZGate indices must be equal or less to the number of levels.',
            )
        self.level_1 = level_1
        self.level_2 = level_2
        
    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        pos = np.exp(1j * params[0] / 2)
        neg = np.exp(-1j * params[0] / 2)
        
        matrix = np.eye(self.num_levels, dtype=np.complex128)*neg
        for level in range(self.num_levels):
            if level!=self.leve_2:
                matrix[self.level_1*self.num_levels+level,self.level_1*self.num_levels+level]=pos
            if level!=self.level_1:
                matrix[level*self.num_levels+self.level_2,level*self.num_levels+self.level_2]=pos

        return UnitaryMatrix(matrix,self.radixes)

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]: #TODO fix
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)

        dpos = 1j / 2 * np.exp(1j * params[0] / 2)
        dneg = -1j / 2 * np.exp(-1j * params[0] / 2)
        
        matrix = np.eye(self.num_levels, dtype=np.complex128)*dneg
        for level in range(self.num_levels):
            if level!=self.leve_2:
                matrix[self.level_1*self.num_levels+level,self.level_1*self.num_levels+level]=dpos
            if level!=self.level_1:
                matrix[level*self.num_levels+self.level_2,level*self.num_levels+self.level_2]=dpos

        return np.array([matrix],dtype=np.complex128)
