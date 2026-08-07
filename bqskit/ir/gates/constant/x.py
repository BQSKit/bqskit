"""This module implements the XGate."""
from __future__ import annotations

from openqudit.expressions import XGate as _XGate

from bqskit.ir.gate import Gate
from bqskit.utils.cachedclass import CachedClass


class XGate(Gate, CachedClass):
    """
    The Pauli X gate.

    The X gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        0 & 1 \\\\
        1 & 0 \\\\
        \\end{pmatrix}
    """

    _num_qudits = 1
    _qasm_name = 'x'
    _expr = _XGate()
