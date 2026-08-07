"""This module implements the CSGate."""
from __future__ import annotations

from openqudit.expressions import Controlled as _Controlled
from openqudit.expressions import SGate as _SGate

from bqskit.ir.gate import Gate
from bqskit.utils.cachedclass import CachedClass


class CSGate(Gate, CachedClass):
    """
    The Controlled-S gate.

    The CS gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & 1 & 0 \\\\
        0 & 0 & 0 & i \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _qasm_name = 'cs'
    _expr = _Controlled(_SGate())
