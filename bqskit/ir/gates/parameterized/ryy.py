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
    A gate representing an arbitrary rotation around the YY axis for qudits.

    
    __init__() arguments:
            num_levels : int
                Number of levels in each qudit (d).
            level_1,level_2, level_3, level_4: int
                The levels on which to apply the Y gates (0...d-1).
    get_unitary arguments:
            param: float
            The angle by which to rotate

    """

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'ryy'

    def __init__(self, num_levels: int=2, level_1: int=0, level_2: int=1, level_3: int=0, level_4: int=1):
        if num_levels <2 or not is_integer(num_levels):
            raise ValueError(
                'YGate num_levels must be a postive integer greater than or equal to 2.',
            )
        self.num_levels = num_levels
        if level_1 > num_levels or level_2 > num_levels or level_3 > num_levels or level_4 > num_levels:
            raise ValueError(
                'YGate indices must be equal or less to the number of levels.',
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
        
        matrix = np.eye(self.num_levels, dtype=np.complex128)
        matrix[self.level_1*self.num_levels+self.level_3,self.level_1*self.num_levels+self.level_3]=cos
        matrix[self.level_1*self.num_levels+self.level_4,self.level_1*self.num_levels+self.level_4]=cos
        matrix[self.level_2*self.num_levels+self.level_3,self.level_2*self.num_levels+self.level_3]=cos
        matrix[self.level_2*self.num_levels+self.level_4,self.level_2*self.num_levels+self.level_4]=cos
        
        matrix[self.level_1*self.num_levels+self.level_3,self.level_2*self.num_levels+self.level_4]=psin
        matrix[self.level_1*self.num_levels+self.level_3,self.level_2*self.num_levels+self.level_4]=nsin
        matrix[self.level_2*self.num_levels+self.level_4,self.level_1*self.num_levels+self.level_3]=nsin
        matrix[self.level_2*self.num_levels+self.level_4,self.level_1*self.num_levels+self.level_3]=psin

        return UnitaryMatrix(matrix,self.radixes)

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)

        dcos = -np.sin(params[0] / 2) / 2
        dnsin = -1j * np.cos(params[0] / 2) / 2
        dpsin = 1j * np.cos(params[0] / 2) / 2
        
        
        matrix = np.zeros((self.num_levels,self.num_levels), dtype=np.complex128)
        matrix[self.level_1*self.num_levels+self.level_3,self.level_1*self.num_levels+self.level_3]=dcos
        matrix[self.level_1*self.num_levels+self.level_4,self.level_1*self.num_levels+self.level_4]=dcos
        matrix[self.level_2*self.num_levels+self.level_3,self.level_2*self.num_levels+self.level_3]=dcos
        matrix[self.level_2*self.num_levels+self.level_4,self.level_2*self.num_levels+self.level_4]=dcos
        
        matrix[self.level_1*self.num_levels+self.level_3,self.level_2*self.num_levels+self.level_4]=dpsin
        matrix[self.level_1*self.num_levels+self.level_3,self.level_2*self.num_levels+self.level_4]=dnsin
        matrix[self.level_2*self.num_levels+self.level_4,self.level_1*self.num_levels+self.level_3]=dnsin
        matrix[self.level_2*self.num_levels+self.level_4,self.level_1*self.num_levels+self.level_3]=dpsin

        return np.array([matrix],dtype=np.complex128)