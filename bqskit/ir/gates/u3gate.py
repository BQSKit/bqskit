from typing import List
from typing import Optional

import numpy as np

from bqskit.ir.qubitgate import QubitGate


class U3Gate(QubitGate):

    num_params = 3
    gate_size = 1

    def get_unitary(self, params: Optional[List[float]] = None) -> np.ndarray:
        return np.zeros(2, 2)  # TODO
