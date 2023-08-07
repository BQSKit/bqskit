"""This module implements the BGate."""
from __future__ import annotations

from numpy import pi
from scipy.linalg import expm
from bqskit.ir.gates.constant.x import XGate
from bqskit.ir.gates.constant.y import YGate
from bqskit.ir.gates.quditgate import QuditGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_integer


class BGate(QuditGate):
    """
    The 2 qudit B gate.

    The B gate is given by the following unitary:

    .. math::
        \\exp(i * \\pi/4 * \\sigma_{xx}) * \\exp(i * \\pi/8 * \\sigma_{yy})

    Unitary expression taken from: https://arxiv.org/pdf/quant-ph/0312193.pdf
    """
    _num_qudits = 2
    _qasm_name = 'b'
    _num_params = 0


    def __init__(
        self, 
        num_levels: int = 2, 
        level_1_1: int = 0, 
        level_1_2: int = 1, 
        level_2_1: int = 0, 
        level_2_2: int = 1,
        level_3_1: int = 0, 
        level_3_2: int = 1, 
        level_4_1: int = 0, 
        level_4_2: int = 1
    ) -> None:
        
        """
            Args:
            num_levels (int): The number of qudit levels (>=2).

            level_1_1, level_1_2 (int): the two levels for the first X qudit gate (<num_levels)
            level_2_1, level_2_2 (int): the two levels for the second X qudit gate (<num_levels)
            level_3_1, level_3_2 (int): the two levels for the first Y qudit gate (<num_levels)
            level_4_1, level_4_2 (int): the two levels for the second Y qudit gate (<num_levels)
            
            Raises:
            ValueError: if num_levels < 2
            ValueError: if any of levels >= num_levels
        """
        if num_levels < 2 or not is_integer(num_levels):
            raise ValueError(
                'X and Y Gate num_levels must be a postive integer greater than or equal to 2.',
            )
        self.num_levels = num_levels
        if level_1_1 > num_levels or level_1_2 > num_levels or level_2_1 > num_levels or level_2_2 > num_levels \
            or level_3_1 > num_levels or level_3_2 > num_levels or level_4_1 > num_levels or level_4_2 > num_levels:
            raise ValueError(
                'X and Y Gate indices must be equal or less to the number of levels.',
            )
        self.level_1_1 = level_1_1
        self.level_1_2 = level_1_2
        self.level_2_1 = level_2_1
        self.level_2_2 = level_2_2
        self.level_3_1 = level_3_1
        self.level_3_2 = level_3_2
        self.level_4_1 = level_4_1
        self.level_4_2 = level_4_2 

    def get_unitary(self) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        x1 = XGate(self._num_levels, self.level_1_1,self.level_1_2).get_unitary()
        x2 = XGate(self._num_levels, self.level_2_1,self.level_2_2).get_unitary()
        y1 = YGate(self._num_levels, self.level_3_1,self.level_3_2).get_unitary()
        y2 = YGate(self._num_levels, self.level_4_1,self.level_4_2).get_unitary()
        xx = x1.otimes(x2)
        yy = y1.otimes(y2)
        return UnitaryMatrix(
            expm(1j * pi / 4 * xx.numpy()) @ expm(1j * pi / 8 * yy.numpy()),
        ) 