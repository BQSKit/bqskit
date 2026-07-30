"""This module implements the YGate."""
from __future__ import annotations

from openqudit.expressions import YGate as _YGate

from bqskit.ir.gate import Gate
from bqskit.utils.cachedclass import CachedClass


class YGate(Gate, CachedClass):
    """
    The Pauli Y gate.

    The Y gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        0 & -i \\\\
        i & 0 \\\\
        \\end{pmatrix}
    """

    _num_qudits = 1
    _qasm_name = 'y'

    _expr = _YGate()
