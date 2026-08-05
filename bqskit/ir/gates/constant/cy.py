"""This module implements the CYGate."""

from __future__ import annotations

from openqudit.expressions import Controlled as _Controlled
from openqudit.expressions import YGate as _YGate

from bqskit.ir.gate import Gate
from bqskit.utils.cachedclass import CachedClass


class CYGate(Gate, CachedClass):
    """
    The Controlled-Y gate.

    The CY gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & 0 & -i \\\\
        0 & 0 & i & 0 \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _qasm_name = 'cy'
    _expr = _Controlled(_YGate())
