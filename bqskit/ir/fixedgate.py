from typing import List
from typing import Optional

import numpy as np

from bqskit.ir.gate import Gate


class FixedGate(Gate):

    num_params = 0

    def get_unitary(self, params: Optional[List[float]] = None) -> np.ndarray:
        if params is not None:
            raise ValueError('Fixed gates do not take parameters.')

        if not hasattr(self.__class__, 'utry'):
            raise AttributeError

        return self.__class__.utry
