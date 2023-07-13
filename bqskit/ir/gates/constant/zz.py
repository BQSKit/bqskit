"""This module implements the ZZGate.""" 
from __future__ import annotations 
import numpy.typing as npt

import math

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.quditgate import QuditGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_integer
from bqskit.ir.gates.constant.z import ZGate
from bqskit.utils.math import kron


class ZZGate(QuditGate):
    """
    The Ising ZZ coupling gate for qutrits
    """

    _num_qudits = 2
    _qasm_name = 'zz'
    
    def __init__(self, num_levels: int=2, level_1: int=0, level_2: int=1):
        if num_levels <2 or not is_integer(num_levels):
            raise ValueError(
                'XGate num_levels must be a postive integer greater than or equal to 2.',
            )
        self.num_levels = num_levels
        if level_1 > num_levels or level_2 > num_levels:
            raise ValueError(
                'XGate indices must be equal or less to the number of levels.',
            )
        self.level_1 = level_1
        self.level_2 = level_2
        
    def get_unitary(self) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""

        return UnitaryMatrix(kron([ZGate(self._num_levels, self.level_1).get_unitary(), 
                       XGate(self._num_levels, self.level_2).get_unitary()]).tolist(), self.radixes)
    
    def get_grad(self) -> npt.NDArray[np.complex128]:
        return np.array([])