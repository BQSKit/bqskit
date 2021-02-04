from __future__ import annotations

from typing import Sequence

import numpy as np

from bqskit.ir.gate import Gate
from bqskit.qis.unitarymatrix import UnitaryMatrix


class FixedGate(Gate):

    num_params = 0

    def get_unitary(self, params: Sequence[float] | None = None) -> UnitaryMatrix:
        if params is not None:
            raise ValueError('Fixed gates do not take parameters.')

        if hasattr(self.__class__, 'utry'):
            return self.__class__.utry

        raise AttributeError(
            'Expected utry class variable for gate %s.'
            % self.__class__.name,
        )
