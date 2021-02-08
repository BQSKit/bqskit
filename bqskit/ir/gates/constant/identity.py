"""This module implements the IdentityGate."""
from __future__ import annotations
from typing import Optional, Sequence

import numpy as np

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.qis.unitarymatrix import UnitaryMatrix

class IdentityGate(ConstantGate):
    """An Identity (No-OP) Gate."""

    def __init__(self, radixes: Optional[Sequence[int]] = None):
        """
        Creates an IdentityGate, defaulting to a single-qubit identity.

        Args:
            radixes (Optional[Sequence[int]]): The number of orthogonal
                states for each qudit.
        """
        self.radixes = radixes or [2]
        self.size = len(radixes)
        self.utry = UnitaryMatrix(np.identity(2**np.prod(self.radixes)))
        self.qasm_name = "identity%d" % self.size
    
    def get_qasm_gate_def(self) -> str:
        """Returns a qasm gate definition block for this gate."""
        param_symbols = ["a%d" % i for i in range(self.size)]
        param_str = ",".join(param_symbols)
        header = "gate identity%d %s" % (self.size, param_str)
        body_stmts = [ "\tU(0,0,0) %s;" % sym for sym in param_symbols ]
        body = "\n".join(body_stmts)
        return "%s\n{\n%s\n}\n" % ( header, body )
