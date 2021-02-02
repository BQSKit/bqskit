from __future__ import annotations

from typing import Sequence

import numpy as np

from bqskit.ir.gate import Gate


class FixedGate(Gate):

    num_params = 0

    def get_unitary(self, params: Sequence[float] | None = None) -> np.ndarray:
        if params is not None:
            raise ValueError('Fixed gates do not take parameters.')

        if not hasattr(self.__class__, 'utry'):
            return self.__class__.utry

        raise AttributeError
