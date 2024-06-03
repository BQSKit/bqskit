"""This module implements the CSDGGate."""
from __future__ import annotations

from bqskit.ir.gate import Gate
from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class CSDGGate(ConstantGate, QubitGate):
    """
    The Controlled-S Dagger gate.

    The CS Dagger gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & 1 & 0 \\\\
        0 & 0 & 0 & -i \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _qasm_name = 'csdg'
    _utry = UnitaryMatrix(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1j],
        ],
    )

    def get_qasm_gate_def(self) -> str:
        """Returns a qasm gate definition block for this gate."""
        return (
            'gate csdg a,b\n'
            '{\n'
            '\tp(-pi/4) a;\n'
            '\tcx a, b;\n'
            '\tp(pi/4) b;\n'
            '\tcx a, b;\n'
            '\tp(-pi/4) b;\n'
            '}\n'
        )

    def get_inverse(self) -> Gate:
        """Return the inverse of this gate."""
        from bqskit.ir.gates.constant.cs import CSGate
        return CSGate()
