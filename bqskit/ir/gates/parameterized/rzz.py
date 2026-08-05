"""This module implements the RZZGate."""

from __future__ import annotations

from openqudit.expressions import RZZGate as _RZZGate

from bqskit.ir.gate import Gate
from bqskit.utils.cachedclass import CachedClass


class RZZGate(
    Gate,
    CachedClass,
):
    """
    A gate representing an arbitrary rotation around the ZZ axis.

    It is given by the following parameterized unitary:

    .. math::

        \\begin{pmatrix}
        \\exp({-i\\frac{\\theta}{2}}) & 0 & 0 & 0 \\\\
        0 & \\exp({i\\frac{\\theta}{2}}) & 0 & 0 \\\\
        0 & 0 & \\exp({i\\frac{\\theta}{2}}) & 0 \\\\
        0 & 0 & 0 & \\exp({-i\\frac{\\theta}{2}}) \\\\
        \\end{pmatrix}
    """

    _qasm_name = 'rzz'
    _expr = _RZZGate()
