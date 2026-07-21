"""This module implements the RZZGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from openqudit.expressions import RZZGate as _RZZGate

from bqskit.ir.gate import Gate
from bqskit.qis.unitary.unitary import RealVector
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

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`~bqskit.ir.gate.Gate` for more info.
        """
        self.check_parameters(params)

        dpos = 1j / 2 * np.exp(1j * params[0] / 2)
        dneg = -1j / 2 * np.exp(-1j * params[0] / 2)

        return np.array(
            [
                [
                    [dneg, 0, 0, 0],
                    [0, dpos, 0, 0],
                    [0, 0, dpos, 0],
                    [0, 0, 0, dneg],
                ],
            ], dtype=np.complex128,
        )
