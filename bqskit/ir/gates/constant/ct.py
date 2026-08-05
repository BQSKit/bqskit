"""This module implements the CTGate."""

from __future__ import annotations

from openqudit.expressions import Controlled as _Controlled
from openqudit.expressions import TGate as _TGate

from bqskit.ir.gate import Gate
from bqskit.utils.cachedclass import CachedClass


class CTGate(Gate, CachedClass):
    """
    The Controlled-T gate.

    The CT gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & 1 & 0 \\\\
        0 & 0 & 0 & \\exp({i\\frac{\\pi}{4}}) \\\\
        \\end{pmatrix}
    """

    _name = 'CTGate'
    _qasm_name = 'ct'
    _expr = _Controlled(_TGate())
