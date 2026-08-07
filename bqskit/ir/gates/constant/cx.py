"""This module implements the CNOTGate/CXGate."""
from __future__ import annotations

from openqudit.expressions import Controlled as _Controlled
from openqudit.expressions import XGate as _XGate

from bqskit.ir.gate import Gate
from bqskit.utils.cachedclass import CachedClass


class CNOTGate(Gate, CachedClass):
    """
    The Controlled-Not or Controlled-X gate.

    The CNOT gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & 0 & 1 \\\\
        0 & 0 & 1 & 0 \\\\
        \\end{pmatrix}
    """

    _name = 'CNOTGate'
    _qasm_name = 'cx'
    _expr = _Controlled(_XGate())


CXGate = CNOTGate
