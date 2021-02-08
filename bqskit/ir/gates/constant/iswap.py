"""This module implements the ISWAPGate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitarymatrix import UnitaryMatrix


class ISWAPGate(ConstantGate, QubitGate):
    """The two qubit swap and phase iSWAP gate."""

    size = 2
    qasm_name = "iswap"
    utry = UnitaryMatrix(
        np.array(
            [
                [1,0,0,0],
                [0,0,1j,0],
                [0,1j,0,0],
                [0,0,0,1],
            ], dtype=np.complex128,
        ),
    )

    def get_qasm_gate_def(self) -> str:
        """Returns a qasm gate definition block for this gate."""
        return "" # TODO
