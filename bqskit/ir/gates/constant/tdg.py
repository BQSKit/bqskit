"""This module implements the TdgGate."""
from __future__ import annotations

from openqudit.expressions import Dagger as _Dagger
from openqudit.expressions import TGate as _TGate

from bqskit.ir.gate import Gate
from bqskit.utils.cachedclass import CachedClass


class TdgGate(Gate, CachedClass):
    """
    The single-qubit T Dagger gate.

    .. math::

        \\begin{pmatrix}
        1 & 0 \\\\
        0 & e^{-i\\frac{\\pi}{4}} \\\\
        \\end{pmatrix}
    """

    _num_qudits = 1
    _qasm_name = 'tdg'
    _expr = _Dagger(_TGate())
