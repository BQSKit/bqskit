from typing import Optional
from typing import Sequence

import numpy as np

from bqskit.ir.gate import Gate


class FixedGate(Gate):

    num_params = 0

    def get_unitary(self, params: Optional[Sequence[float]] = None) -> np.ndarray:
        if params is not None:
            raise ValueError('Fixed gates do not take parameters.')

        if not hasattr(self.__class__, 'utry'):
            return self.__class__.utry

        raise AttributeError
