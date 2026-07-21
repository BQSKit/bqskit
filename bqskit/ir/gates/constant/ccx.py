"""This module implements the CCXGate/ToffoliGate."""
from __future__ import annotations

from openqudit.expressions import Controlled as _Controlled
from openqudit.expressions import XGate as _XGate

from bqskit.ir.gate import Gate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class CCXGate(Gate, CachedClass):
    """
    The toffoli gate, equal to an X gate with two controls.

    The Toffoli gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\\\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\\\
        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\\\
        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\\\
        \\end{pmatrix}
    """

    _num_qudits = 3
    _qasm_name = 'ccx'
    _expr = _Controlled(_XGate(), [2, 2])
    _utry = UnitaryMatrix(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ],
    )


ToffoliGate = CCXGate
