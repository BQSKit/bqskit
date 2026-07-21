"""This module implements the RYYGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from openqudit.expressions import RYYGate as _RYYGate

from bqskit.ir.gate import Gate
from bqskit.qis.unitary.unitary import RealVector
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

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`~bqskit.ir.gate.Gate` for more info.
        """
        self.check_parameters(params)

        dcos = -np.sin(params[0] / 2) / 2
        dnsin = -1j * np.cos(params[0] / 2) / 2
        dpsin = 1j * np.cos(params[0] / 2) / 2

        return np.array(
            [
                [
                    [dcos, 0, 0, dpsin],
                    [0, dcos, dnsin, 0],
                    [0, dnsin, dcos, 0],
                    [dpsin, 0, 0, dcos],
                ],
            ], dtype=np.complex128,
        )
