"""This module implements the YYGate."""  # TODO adapt for qudits
from __future__ import annotations

import numpy.typing as npt

from bqskit.ir.gates.constant.y import YGate
from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.quditgate import QuditGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.math import kron
from bqskit.utils.typing import is_integer


class YYGate(QuditGate):
    """
    The Ising YY coupling gate for qudits.

    __init__() arguments:
            num_levels : int
                Number of levels in each qudit (d).
            level_1,level_2, level_3, lelvel_4: int
                The levels on which to apply the Y gate (0...d-1).
    """

    _num_qudits = 2
    _qasm_name = 'yy'
    _num_params = 0

    def __init__(self, num_levels: int = 2, level_1: int = 0, level_2: int = 1, level_3: int = 0, level_4: int = 1):
        if num_levels < 2 or not is_integer(num_levels):
            raise ValueError(
                'YYGate num_levels must be a postive integer greater than or equal to 2.',
            )
        self.num_levels = num_levels
        if level_1 > num_levels or level_2 > num_levels or level_3 > num_levels or level_4 > num_levels:
            raise ValueError(
                'YYGate indices must be equal or less to the number of levels.',
            )
        self.level_1 = level_1
        self.level_2 = level_2
        self.level_3 = level_3
        self.level_4 = level_4

    def get_unitary(self) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""

        return UnitaryMatrix(
            kron([
                YGate(
                    self._num_levels, self.level_1,
                    self.level_2,
                ).get_unitary(),
                YGate(
                    self._num_levels, self.level_3,
                    self.level_4,
                ).get_unitary(),
            ]).tolist(), self.radixes,
        )

    def get_grad(self) -> npt.NDArray[np.complex128]:
        return np.array([])
