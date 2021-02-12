"""This module implements the ISwapGate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitarymatrix import UnitaryMatrix


class ISwapGate(ConstantGate, QubitGate):
    """The two qubit swap and phase iSWAP gate."""

    size = 2
    qasm_name = "iswap"
    utry = UnitaryMatrix(
        [
            [1,0,0,0],
            [0,0,1j,0],
            [0,1j,0,0],
            [0,0,0,1],
        ]
    )

    def get_qasm_gate_def(self) -> str:
        """Returns a qasm gate definition block for this gate."""
        return ( "gate iswap a,b\n"
                 "{\n"
                 "\ts a;\n"
                 "\ts b;\n"
                 "\th a;\n"
                 "\tcx a, b;\n"
                 "\tcx b, a;\n"
                 "\th b;\n"
                 "}\n" )
