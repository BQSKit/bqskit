"""This module implements the ZZGate."""
from __future__ import annotations

from bqskit.ir.gates.constant.z import ZGate
from bqskit.ir.gates.quditgate import QuditGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_integer


class ZZGate(QuditGate):
    """The Ising ZZ coupling gate for qutrits."""

    _num_qudits = 2
    _qasm_name = 'zz'

    def __init__(
        self, 
        num_levels: int = 2, 
        level_1: int = 0, 
        level_2: int = 1,
        level_3: int = 0, 
        level_4: int = 1, 
    ) -> None:
        """
            Args:
            num_levels (int): The number of qudit levels (>=2).

            level_1 (int): the first level for the first Z qudit gate (<num_levels)
            level_2 (int): the second level for the first Z qudit gate (<num_levels)
            level_3 (int): the first level for the second Z qudit gate (<num_levels)
            level_4 (int): the second level for the second Z qudit gate (<num_levels) 
            
            Raises:
            ValueError: if num_levels < 2
            ValueError: if any of levels >= num_levels
        """
        if num_levels < 2 or not is_integer(num_levels):
            raise ValueError(
                'XGate num_levels must be a postive integer greater than or equal to 2.',
            )
        self.num_levels = num_levels
        if level_1 > num_levels or level_2 > num_levels or level_3 > num_levels or level_4 > num_levels:
            raise ValueError(
                'ZZGate indices must be equal or less to the number of levels.',
            )
        self.level_1 = level_1
        self.level_2 = level_2
        self.level_3 = level_3
        self.level_4 = level_4

    def get_unitary(self) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        z1 = ZGate(self._num_levels, self.level_1, self.level_2).get_unitary()
        z2 = ZGate(self._num_levels, self.level_3, self.level_4).get_unitary()
        return z1.otimes(z2)