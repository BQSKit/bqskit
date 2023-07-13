"""This module implements the RXGate.""" #Done
from __future__ import annotations

import numpy as np
from bqskit.ir.gates.quditgate import QuditGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass
from bqskit.utils.typing import is_integer
 
    
class RXGate(
    QuditGate,
    DifferentiableUnitary,
    CachedClass,
):
    """
        A gate representing an arbitrary rotation by the X qudit gate 
        This is equivalent to rotation by the X Pauli Gate in the subspace of 2 levels
    
        __init__() arguments:
            num_levels : int
                Number of levels in each qudit (d).
            level_1,level_2: int
                 The levels on which to apply the X gate (0...d-1).
                        
        get_unitary arguments:
                param: float
                The angle by which to rotate

    """ 

    _num_qudits = 1
    _num_params = 1
    _qasm_name = 'rx'
    
    def __init__(self, num_levels: int=2, level_1: int=0, level_2: int=1):
        """
            Raises:
            ValueError: If `num_levels` is less than 2 or not a positive integer
                        If level >= num_levels
                        IF Gate.radixes != num_levels
        """
        if num_levels <2 or not is_integer(num_levels):
            raise ValueError(
                'RXGate num_levels must be a postive integer greater than or equal to 2.',
            )
        self.num_levels = num_levels
        if level_1 > num_levels or level_2 > num_levels:
            raise ValueError(
                'XGate indices must be equal or less to the number of levels.',
            )
        self.level_1 = level_1
        self.level_2 = level_2

   
    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        
        self.check_parameters(params)
        
        cos = np.cos(params[0] / 2)
        sin = -1j * np.sin(params[0] / 2)
        
        matrix = np.eye(self.num_levels, dtype=np.complex128)
        matrix[self.level_1,self.level_1] = cos
        matrix[self.level_2,self.level_2] = cos
        matrix[self.level_1,self.level_2] = sin
        matrix[self.level_2,self.level_1] = sin
        
        return UnitaryMatrix(matrix, self.radixes)
    
            
       
    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)
        
        dcos = -np.sin(params[0] / 2) / 2
        dsin = -1j * np.cos(params[0] / 2) / 2
        
        matrix = np.zeros((self.num_levels, self.num_levels), dtype=np.complex128)
        matrix[self.level_1,self.level_1] = dcos
        matrix[self.level_2,self.level_2] = dcos
        matrix[self.level_1,self.level_2] = dsin
        matrix[self.level_2,self.level_1] = dsin
        
        return np.array([matrix],dtype=np.complex128)