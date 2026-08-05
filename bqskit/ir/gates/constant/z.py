"""This module implements the ZGate."""

from __future__ import annotations

from openqudit.expressions import ZGate as _ZGate

from bqskit.ir.gate import Gate
from bqskit.utils import CachedClass


class ZGate(Gate, CachedClass):
    """
    The Pauli Z gate.

    The Z gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 \\\\
        0 & -1 \\\\
        \\end{pmatrix}
    """

    _num_qudits = 1
    _qasm_name = 'z'
    _expr = _ZGate()
