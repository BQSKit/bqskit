"""This module implements the SqrtCNOTDGGate."""
from __future__ import annotations

from bqskit.ir.gate import Gate
from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class SqrtCNOTDGGate(ConstantGate, QubitGate):
    """
    The Square root Controlled-X gate.

    The SqrtCNOT Dagger gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & \\frac{1}{2} - \\frac{1}{2}i & \\frac{1}{2} + \\frac{1}{2}i \\\\
        0 & 0 & \\frac{1}{2} + \\frac{1}{2}i & \\frac{1}{2} - \\frac{1}{2}i \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _qasm_name = 'csxdg'
    _utry = UnitaryMatrix(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0.5 - 0.5j, 0.5 + 0.5j],
            [0, 0, 0.5 + 0.5j, 0.5 - 0.5j],
        ],
    )

    def get_qasm_gate_def(self) -> str:
        """Returns a qasm gate definition block for this gate."""
        """Gate csxdg q0,q1 { u(0,0,pi/4) q1; cx q0,q1; u(0,0,-pi/4) q1; cx
        q0,q1; cu(pi/2,0,pi,0) q0,q1; u(0,0,pi/4) q1; cx q0,q1; u(0,0,-pi/4) q1;
        cx q0,q1; p(pi/4) q0; }"""
        return (
            'gate csxdg a,b\n'
            '{\n'
            '\tu(0,0,pi/4) b;\n'
            '\tcx a,b;\n'
            '\tu(0,0,-pi/4) b;\n'
            '\tcx a,b;\n'
            '\tcu(pi/2,0,pi,0) a,b;\n'
            '\tu(0,0,pi/4) b;\n'
            '\tcx a,b;\n'
            '\tu(0,0,-pi/4) b;\n'
            '\tcx a,b;\n'
            '\tp(pi/4) a;\n'
            '}\n'
        )

    def get_inverse(self) -> Gate:
        """Return the inverse of this gate."""
        from bqskit.ir.gates.constant.sqrtcnot import SqrtCNOTGate
        return SqrtCNOTGate()
