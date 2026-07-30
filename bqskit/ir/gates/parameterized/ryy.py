"""This module implements the RYYGate."""
from __future__ import annotations

from openqudit.expressions import RYYGate as _RYYGate

from bqskit.ir.gate import Gate
from bqskit.utils.cachedclass import CachedClass


class RYYGate(
    Gate,
    CachedClass,
):
    """
    A gate representing an arbitrary rotation around the YY axis.

    It is given by the following parameterized unitary:

    .. math::

        \\begin{pmatrix}
        \\cos{\\frac{\\theta}{2}} & 0 & 0 & \\sin{\\frac{\\theta}{2}}i \\\\
        0 & \\cos{\\frac{\\theta}{2}} & -\\sin{\\frac{\\theta}{2}}i & 0 \\\\
        0 & -\\sin{\\frac{\\theta}{2}}i & \\cos{\\frac{\\theta}{2}} & 0 \\\\
        \\sin{\\frac{\\theta}{2}}i & 0 & 0 & \\cos{\\frac{\\theta}{2}} \\\\
        \\end{pmatrix}
    """

    _qasm_name = 'ryy'
    _expr = _RYYGate()
