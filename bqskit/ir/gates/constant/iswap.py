"""This module implements the ISwapGate."""
from __future__ import annotations

from openqudit.expressions import UnitaryExpression as _UnitaryExpression

from bqskit.ir.gate import Gate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class ISwapGate(Gate, CachedClass):
    """
    The two qubit swap and phase iSWAP gate.

    The ISwap gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 0 & i & 0 \\\\
        0 & i & 0 & 0 \\\\
        0 & 0 & 0 & 1 \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _qasm_name = 'iswap'
    _expr = _UnitaryExpression(
        'ISwap() { [[1,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,1]] }',
    )
    _utry = UnitaryMatrix(
        [
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1],
        ],
    )

    def get_qasm_gate_def(self) -> str:
        """Returns a qasm gate definition block for this gate."""
        return (
            'gate iswap a,b\n'
            '{\n'
            '\ts a;\n'
            '\ts b;\n'
            '\th a;\n'
            '\tcx a, b;\n'
            '\tcx b, a;\n'
            '\th b;\n'
            '}\n'
        )
