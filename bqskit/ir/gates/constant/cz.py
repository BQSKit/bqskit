"""This module implements the CZGate."""
from __future__ import annotations

from openqudit.expressions import Controlled as _Controlled
from openqudit.expressions import ZGate as _ZGate

from bqskit.ir.gate import Gate
from bqskit.utils.cachedclass import CachedClass


class CZGate(Gate, CachedClass):
    """
    The Controlled-Z gate.

    The CZ gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & 1 & 0 \\\\
        0 & 0 & 0 & -1 \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _qasm_name = 'cz'
    _expr = _Controlled(_ZGate())
