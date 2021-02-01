"""This module implements the CNOT Gate."""
import numpy as np

from bqskit.ir.fixedgate import FixedGate
from bqskit.ir.qubitgate import QubitGate


class CNOTGate(FixedGate, QubitGate):

    gate_size = 2
    utry = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=np.complex128,
    )
