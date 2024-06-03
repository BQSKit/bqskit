"""This module implements the SqrtISwapDGGate."""
from __future__ import annotations

import math

from bqskit.ir.gate import Gate
from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class SqrtISwapDGGate(ConstantGate, QubitGate):
    """
    The square root two qubit swap and phase iSWAP Dagger gate.

    The SqrtISwap Dagger gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & \\frac{1}{\\sqrt{2}} & - \\frac{i}{\\sqrt{2}} & 0 \\\\
        0 & - \\frac{i}{\\sqrt{2}} & \\frac{1}{\\sqrt{2}} & 0 \\\\
        0 & 0 & 0 & 1 \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _qasm_name = 'sqiswdg'
    _utry = UnitaryMatrix(
        [
            [1, 0, 0, 0],
            [0, 1 / math.sqrt(2), -1j / math.sqrt(2), 0],
            [0, -1j / math.sqrt(2), 1 / math.sqrt(2), 0],
            [0, 0, 0, 1],
        ],
    )

    def get_qasm_gate_def(self) -> str:
        """Returns a qasm gate definition block for this gate."""
        return (
            'gate sqiswdg a,b\n'
            '{\n'
            '\tcx a,b;\n'
            '\th a;\n'
            '\tt a;\n'
            '\tcx b,a;\n'
            '\ttdg a;\n'
            '\tcx b,a;\n'
            '\th a;\n'
            '\tcx a,b;\n'
            '}\n'
        )

    def get_inverse(self) -> Gate:
        """Return the inverse of this gate."""
        from bqskit.ir.gates.constant.sqrtiswap import SqrtISwapGate
        return SqrtISwapGate()
