"""This module implements the RZGate."""

from __future__ import annotations

from openqudit.expressions import RZGate as _RZGate

from bqskit.ir.gate import Gate
from bqskit.utils.cachedclass import CachedClass


class RZGate(Gate, CachedClass):
    """
    A gate representing an arbitrary rotation around the Z axis.

    It is given by the following parameterized unitary:

    .. math::

        \\begin{pmatrix}
        \\exp({-i\\frac{\\theta}{2}}) & 0 \\\\
        0 & \\exp({i\\frac{\\theta}{2}}) \\\\
        \\end{pmatrix}
    """

    _qasm_name = 'rz'
    _expr = _RZGate()
