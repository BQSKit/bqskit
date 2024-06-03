"""This module implements the ISwapDGGate."""
from __future__ import annotations

from bqskit.ir.gate import Gate
from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class ISwapDGGate(ConstantGate, QubitGate):
    """
    The two qubit swap and phase iSWAP Dagger gate.

    The ISwap Dagger gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 0 & -i & 0 \\\\
        0 & -i & 0 & 0 \\\\
        0 & 0 & 0 & 1 \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _qasm_name = 'iswap_dg'
    _utry = UnitaryMatrix(
        [
            [1, 0, 0, 0],
            [0, 0, -1j, 0],
            [0, -1j, 0, 0],
            [0, 0, 0, 1],
        ],
    )

    def get_qasm_gate_def(self) -> str:
        """Returns a qasm gate definition block for this gate."""
        """iswap_dg q0,q1 { h q1; cx q1,q0; cx q0,q1; h q0; sdg q1; sdg q0; }"""
        return (
            'gate iswap_dg a,b\n'
            '{\n'
            '\th b;\n'
            '\tcx b, a;\n'
            '\tcx a, b;\n'
            '\th a;\n'
            '\tsdg b;\n'
            '\tsdg a;\n'
            '}\n'
        )

    def get_inverse(self) -> Gate:
        """Return the inverse of this gate."""
        from bqskit.ir.gates.constant.iswap import ISwapGate
        return ISwapGate()
